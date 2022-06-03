import torch
import bmtrain as bmt
import numpy as np
import html
from transformers import BertTokenizer

from .cnewsum_dataset import DyleDemoDataset, dyle_collate_fn
from .retriever_model import RetrieverModel
from .generator_model import GeneratorModel
from model_center.dataset import DistributedDataLoader
from model_center.tokenizer import CPM1Tokenizer

from .generation import generate
from .config import Config


class DyleInfer:
    def __init__(self):
        bmt.init_distributed(loss_scale_factor=2, loss_scale_steps=1024)
        self.retriever_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.generator_tokenizer = CPM1Tokenizer.from_pretrained(Config.model_path)
        self.retriever_model = RetrieverModel().cuda()
        self.generator_model = GeneratorModel(Config.model_path, Config.top_k, Config.max_source_len, Config.max_target_len).cuda()
        
        self.load_model(Config.name)

    def load_model(self, name):
        bmt.load(self.retriever_model, Config.save_model_dir + name + 'retriever_model.pt')
        bmt.load(self.generator_model, Config.save_model_dir + name + 'generator_model.pt')
        
    def get_summary(self, str_list: list):
        dataset = DyleDemoDataset(str_list, self.retriever_tokenizer, self.generator_tokenizer)
        dataloader = DistributedDataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dyle_collate_fn)
        
        self.retriever_model.eval()
        self.generator_model.eval()
        
        outputs = []
        with torch.no_grad():
            for iter_num, (id, text_sents, para_pos, retriever_input_ids, retriever_attention_masks, cls_ids, context_input_ids) in enumerate(dataloader):
                retriever_input_ids, retriever_attention_masks, cls_ids, context_input_ids, = \
                    retriever_input_ids.cuda().squeeze(0), retriever_attention_masks.cuda().squeeze(0), cls_ids.cuda().squeeze(0), context_input_ids.cuda().squeeze(0)

                retriever_outputs = self.retriever_model(input_ids=retriever_input_ids.cuda(), attention_mask=retriever_attention_masks.cuda())
                retriever_cls_logits = retriever_outputs.contiguous().view(-1)[cls_ids.cpu().tolist()]

                _, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(Config.top_k, retriever_cls_logits.shape[0]))
                initial_topk_indices = retriever_topk_indices = retriever_topk_indices.cpu().tolist()
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

                # output
                for j in range(len(text_sents)):
                    for pos in para_pos[j]:
                        text_sents[j][pos] = text_sents[j][pos] + "\n"
                    for i in range(len(text_sents[j])):
                        text_sents[j][i] = html.escape(text_sents[j][i])
                
                # only works for batch size = 1
                initial_topk_indices = sorted(initial_topk_indices)
                text_lens = np.array([0] + [len(text) for text in text_sents[0]])
                cum_text_lens = np.cumsum(text_lens)
                highlight_spans = []
                for idx in initial_topk_indices:
                    highlight_spans.append({"start": cum_text_lens[idx], "end": cum_text_lens[idx + 1]})
                outputs.append({
                    "paragraph": "".join(text_sents[0]),
                    "summary": output[0],
                    "index": highlight_spans
                })

        return outputs
