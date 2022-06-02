import torch, os, re
from torch.distributions.categorical import Categorical
from torch.distributions import kl_divergence
import bmtrain as bmt
import numpy as np
from transformers import BertTokenizer

from cnewsum_dataset import CNewSumDataset
from retriever_model import RetrieverModel
from generator_model import GeneratorModel
from model_center.dataset import DistributedDataLoader
from model_center.tokenizer import CPM1Tokenizer

from generation import generate
from config import Config
from rouge import Rouge


class FineTuneCPM:

    def __init__(self):
        bmt.init_distributed(loss_scale_factor=2, loss_scale_steps=1024)
        if Config.input_cache_dir != None:
            os.makedirs(Config.input_cache_dir, exist_ok=True)
        if Config.save_model_dir != None:
            os.makedirs(Config.save_model_dir, exist_ok=True)
        if Config.output_dir != None:
            os.makedirs(Config.output_dir, exist_ok=True)

        self.retriever_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.generator_tokenizer = CPM1Tokenizer.from_pretrained(Config.model_path)
        self.retriever_model = RetrieverModel().cuda()
        self.generator_model = GeneratorModel(Config.model_path, Config.top_k, Config.max_source_len, Config.max_target_len).cuda()
        bmt.synchronize()
        self.retriever_optimizer = bmt.optim.AdamOptimizer(self.retriever_model.parameters(), lr=Config.retriever_start_lr)
        self.generator_optimizer = bmt.optim.AdamOptimizer(self.generator_model.parameters(), lr=Config.generator_start_lr)
        self.retriever_lr, self.generator_lr = Config.retriever_start_lr, Config.generator_start_lr
        bmt.synchronize()

    def prepare_dataset(self):
        splits = ['train', 'dev']
        self.datasets = {}
        for split in splits:
            self.datasets[split] = CNewSumDataset(Config.data_path, split, self.retriever_tokenizer, self.generator_tokenizer)
    
    def load_model(self, name):
        bmt.load(self.retriever_model, Config.save_model_dir + name + '_retriever_model.pt')
        bmt.load(self.generator_model, Config.save_model_dir + name + '_generator_model.pt')

    def train(self):
        criterion_retriever = torch.nn.CrossEntropyLoss(reduction='none')
        # criterion_generator = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        best_metrics = -float('inf')
        epoch_num, decay_num, no_improvement_num = 0, 0, 0
        dataloader = {
            "train": DistributedDataLoader(self.datasets['train'], batch_size=1, shuffle=True),
            "dev": DistributedDataLoader(self.datasets['dev'], batch_size=1, shuffle=False),
        }

        while decay_num < Config.max_decay_num:
            epoch_num += 1
            self.retriever_model.train()
            self.generator_model.train()
            self.retriever_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            # train
            bmt.print_rank(f'Training epoch {epoch_num}.')
            for iter_num, [retriever_input_ids, retriever_attention_masks, cls_ids, oracle, context_input_ids, labels] in enumerate(dataloader['train']):
                retriever_input_ids, retriever_attention_masks, cls_ids, oracle, context_input_ids, labels = \
                    retriever_input_ids.cuda().squeeze(0), retriever_attention_masks.cuda().squeeze(0), cls_ids.cuda().squeeze(0), oracle.cuda().squeeze(0), context_input_ids.cuda().squeeze(0), labels.cuda().squeeze(0)

                retriever_outputs = self.retriever_model(input_ids=retriever_input_ids.cuda(), attention_mask=retriever_attention_masks.cuda())
                retriever_cls_logits = retriever_outputs.contiguous().view(-1)[cls_ids.cpu().tolist()]  # [NUM_OF_SENTENCE]

                # retriever loss
                retriever_loss = 0
                for sentence_id in oracle.cpu().tolist():
                    retriever_loss = retriever_loss + criterion_retriever(input=retriever_cls_logits.unsqueeze(0), target=torch.LongTensor([sentence_id]).cuda())
                if len(oracle) > 0:
                    retriever_loss = retriever_loss / len(oracle)

                if oracle.shape[0] != 0:
                    k = min(Config.top_k, retriever_cls_logits.shape[0])
                    # use oracle
                    retriever_topk_indices = oracle.cpu().tolist()[:k]
                    # if oracle less than k, fill retriever_top_k_indices
                    if len(retriever_topk_indices) < k:
                        _, real_retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(Config.top_k, retriever_cls_logits.shape[0]))
                        real_retriever_topk_indices = real_retriever_topk_indices.cpu().tolist()
                        retriever_topk_indices = retriever_topk_indices + [idx for idx in real_retriever_topk_indices if idx not in retriever_topk_indices]
                        retriever_topk_indices = retriever_topk_indices[:k]
                    doc_scores = retriever_cls_logits[retriever_topk_indices]
                else:
                    doc_scores, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(Config.top_k, retriever_cls_logits.shape[0]))
                    retriever_topk_indices = retriever_topk_indices.cpu().tolist()

                # if sentence number less than k, fill retriever_top_k_indices
                if len(retriever_topk_indices) < Config.top_k:
                    doc_scores = torch.cat([doc_scores, torch.zeros((Config.top_k - len(retriever_topk_indices))).fill_(-float('inf')).cuda()])
                    retriever_topk_indices = retriever_topk_indices + [retriever_topk_indices[-1]] * (Config.top_k - len(retriever_topk_indices))

                # remove padding
                labels = labels[:len(torch.nonzero(labels))]
                # shift left
                labels_truncated = labels[:-1]
                input_tokens = torch.cat((context_input_ids[retriever_topk_indices].contiguous().view(-1), labels_truncated)).int()
                input_length = Config.top_k * Config.max_source_len + len(labels_truncated)
                input_context = torch.Tensor([1] * Config.top_k * Config.max_source_len + [0] * len(labels_truncated)).cuda().int()
                input_span = torch.zeros(input_length).int()
                logits, dynamic_scores, _ = self.generator_model(input_tokens.unsqueeze(0), torch.Tensor([input_length]).cuda().unsqueeze(0), input_context.unsqueeze(0), input_span.unsqueeze(0))
                logits, dynamic_scores = logits.squeeze(0), dynamic_scores.squeeze(0)

                labels = labels.unsqueeze(-1)  # (max_target_len, 1)
                generation_loss = logits.gather(dim=-1, index=labels)  # (max_target_len, 1)
                generation_loss.masked_fill_(labels.eq(self.generator_tokenizer.pad_id), 0.0)
                generation_loss = -generation_loss.sum(0).squeeze(0)

                # calc consistency loss
                g_probs = torch.softmax(dynamic_scores, dim=0).mean(1)
                g_probs = g_probs.detach()
                r_probs = torch.softmax(doc_scores, dim=0).clamp(min=1e-5)
                consistency_loss = kl_divergence(Categorical(probs=g_probs), Categorical(probs=r_probs))

                # total loss
                tot_loss = retriever_loss * Config.retriever_alpha + generation_loss * Config.generation_alpha + consistency_loss * Config.consistency_alpha
                tot_loss = tot_loss / Config.gradient_accumulation_steps
                tot_loss.backward()

                if (iter_num + 1) % Config.gradient_accumulation_steps == 0:
                    bmt.optim.clip_grad_norm(self.retriever_optimizer.param_groups, Config.max_grad_norm, scale=self.retriever_optimizer.scale, norm_type=2)
                    bmt.optim.clip_grad_norm(self.generator_optimizer.param_groups, Config.max_grad_norm, scale=self.generator_optimizer.scale, norm_type=2)
                    bmt.optim_step(self.retriever_optimizer)
                    bmt.optim_step(self.generator_optimizer)
                    self.retriever_optimizer.zero_grad()
                    self.generator_optimizer.zero_grad()
                if (iter_num + 1) % Config.save_steps == 0:
                    bmt.save(self.retriever_model, Config.save_model_dir + 'training_' + str(epoch_num) + '_' + str(iter_num) + '_retriever_model.pt')
                    bmt.save(self.generator_model, Config.save_model_dir + 'training_' + str(epoch_num) + '_' + str(iter_num) + '_generator_model.pt')
                if (iter_num + 1) % Config.train_log_steps == 0:
                    bmt.print_rank(f'Training iter {iter_num + 1}.')

            # validation
            self.retriever_model.eval()
            self.generator_model.eval()
            with torch.no_grad():
                inputs, selections, outputs, refs = list(), list(), list(), list()
                bmt.print_rank(f'Validate epoch {epoch_num}.')
                for iter_num, [retriever_input_ids, retriever_attention_masks, cls_ids, oracle, context_input_ids, labels] in enumerate(dataloader['dev']):
                    retriever_input_ids, retriever_attention_masks, cls_ids, oracle, context_input_ids, labels = \
                        retriever_input_ids.cuda().squeeze(0), retriever_attention_masks.cuda().squeeze(0), cls_ids.cuda().squeeze(0), oracle.cuda().squeeze(0), context_input_ids.cuda().squeeze(0), labels.cuda().squeeze(0)

                    retriever_outputs = self.retriever_model(input_ids=retriever_input_ids.cuda(), attention_mask=retriever_attention_masks.cuda())
                    retriever_cls_logits = retriever_outputs.contiguous().view(-1)[cls_ids.cpu().tolist()]

                    _, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(Config.top_k, retriever_cls_logits.shape[0]))
                    retriever_topk_indices = retriever_topk_indices.cpu().tolist()
                    if len(retriever_topk_indices) < Config.top_k:
                        retriever_topk_indices = retriever_topk_indices + [retriever_topk_indices[-1]] * (Config.top_k - len(retriever_topk_indices))

                    input_tokens = torch.cat((context_input_ids[retriever_topk_indices].contiguous().view(-1), torch.zeros(Config.max_target_len).cuda())).int()
                    source_length = Config.top_k * Config.max_source_len
                    input_context = torch.Tensor([1] * Config.top_k * Config.max_source_len + [0] * Config.max_target_len).cuda().int()
                    input_span = torch.zeros_like(input_context).int()

                    # infer
                    input_dict = { 'input_tokens': input_tokens.unsqueeze(0), 'input_span': input_span.unsqueeze(0), 'context': input_context.unsqueeze(0), 'source_length': source_length }
                    output = generate(self.generator_model, self.generator_tokenizer, input_dict, beam=Config.infer_beam_size, temperature=Config.infer_temperature,
                        length_penalty=Config.infer_length_penalty, top_k=Config.infer_top_k, top_p=Config.infer_top_p, no_repeat_ngram_size=Config.infer_no_repeat_ngram_size,
                        repetition_penalty=Config.infer_repetition_penalty, random_sample=Config.infer_random_sample, min_len=2)

                    # save results
                    inputs.append(self.generator_tokenizer.decode(context_input_ids.contiguous().view(-1).cpu().tolist()) + '\n')
                    selections.append(self.generator_tokenizer.decode(context_input_ids[retriever_topk_indices].contiguous().view(-1).cpu().tolist()) + '\n')
                    outputs.append(output[0] + '\n')
                    refs.append(self.generator_tokenizer.decode(labels.cpu().tolist()) + '\n')
                    if (iter_num + 1) % Config.validation_log_steps == 0:
                        bmt.print_rank(f'Validating iter {iter_num + 1}.')

                with open(Config.output_dir + str(bmt.rank()) + '_' + str(epoch_num) + '_inputs.txt', 'w') as fp:
                    fp.writelines(inputs)
                with open(Config.output_dir + str(bmt.rank()) + '_' + str(epoch_num) + '_selections.txt', 'w') as fp:
                    fp.writelines(selections)
                with open(Config.output_dir + str(bmt.rank()) + '_' + str(epoch_num) + '_outputs.txt', 'w') as fp:
                    fp.writelines(outputs)
                with open(Config.output_dir + str(bmt.rank()) + '_' + str(epoch_num) + '_refs.txt', 'w') as fp:
                    fp.writelines(refs)

                rouge_1, rouge_2, rouge_l = self.rouge_score(outputs, refs)
                global_rouge_1 = bmt.sum_loss(torch.Tensor([rouge_1]).cuda()).item()
                global_rouge_2 = bmt.sum_loss(torch.Tensor([rouge_2]).cuda()).item()
                global_rouge_l = bmt.sum_loss(torch.Tensor([rouge_l]).cuda()).item()
                bmt.print_rank(f'Rouge score: rouge-1 {global_rouge_1}, rouge-2 {global_rouge_2}, rouge-l {global_rouge_l}.')
                if global_rouge_1 + global_rouge_2 + global_rouge_l > best_metrics:
                    # save best model
                    bmt.print_rank(f'Update metrics: {best_metrics} -> {global_rouge_1 + global_rouge_2 + global_rouge_l}.')
                    best_metrics = global_rouge_1 + global_rouge_2 + global_rouge_l
                    no_improvement_num = 0
                    bmt.save(self.retriever_model, Config.save_model_dir + 'best_' + str(epoch_num) + '_retriever_model.pt')
                    bmt.save(self.generator_model, Config.save_model_dir + 'best_' + str(epoch_num) + '_generator_model.pt')
                else:
                    # no improvement
                    no_improvement_num += 1
                    if no_improvement_num == Config.no_improvement_decay:
                        # reduce learning rate
                        decay_num += 1
                        self.retriever_lr, self.generator_lr = self.retriever_lr / 2, self.generator_lr / 2
                        for param_group in self.retriever_optimizer.param_groups:
                            param_group['lr'] = self.retriever_lr
                        for param_group in self.retriever_optimizer.param_groups:
                            param_group['lr'] = self.generator_lr
                        no_improvement_num = 0
                    bmt.print_rank(f'No improvement: no improvement {no_improvement_num} / {Config.no_improvement_decay}, decay num {decay_num} / {Config.max_decay_num}.')

    def rouge_score(self, preds, refs):
        rouge_1, rouge_2, rouge_l = list(), list(), list()
        for pred, ref in zip(preds, refs):
            pred = re.sub('<\w+>', '', pred)
            ref = re.sub('<\w+>', '', ref)
            pred = ' '.join(pred)
            ref = ' '.join(ref)

            if len(ref) == 0 and len(pred) == 0:
                continue
            elif len(pred) == 0:
                rouge_1.append(0)
                rouge_2.append(0)
                rouge_l.append(0)
            else:
                score = Rouge().get_scores(refs=ref, hyps=pred)[0]
                rouge_1.append(score['rouge-1']['f'])
                rouge_2.append(score['rouge-2']['f'])
                rouge_l.append(score['rouge-l']['f'])

        return np.array(rouge_1).mean(), np.array(rouge_2).mean(), np.array(rouge_l).mean()


if __name__ == "__main__":
    fine_tune_cpm = FineTuneCPM()
    fine_tune_cpm.prepare_dataset()
    fine_tune_cpm.load_model('best_1')
    fine_tune_cpm.train()
