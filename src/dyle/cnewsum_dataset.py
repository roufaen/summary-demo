from pyparsing import line
import torch, json, os
import bmtrain as bmt
import re
import spacy
from typing import Tuple, List, Dict

from config import Config


class CNewSumDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, split, retriever_tokenizer, generator_tokenizer) -> None:
        self.datas = list()
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_tokenizer = generator_tokenizer
        file = data_path + split + '.simple.label.jsonl'
        bmt.print_rank(f'Start loading dataset {file}.')
        if os.path.exists(Config.input_cache_dir + split + '.json') == False:
            with open(file, 'r') as fp:
                for i, line in enumerate(fp):
                    if (i + 1) % 10000 == 0:
                        bmt.print_rank(f'Loading dataset number {i + 1}.')
                    line_json = json.loads(line)
                    retriever_input_ids, retriever_attention_masks, cls_ids = self.tokenize_retriever(src_text=line_json['article'])
                    context_input_ids, labels = self.tokenize_generator(src_text=line_json['article'], summary=line_json['summary'])

                    self.datas.append({
                        'retriever_input_ids': retriever_input_ids,
                        'retriever_attention_masks': retriever_attention_masks,
                        'cls_ids': cls_ids,
                        'oracle': line_json['label'],
                        'context_input_ids': context_input_ids,
                        'labels': labels
                    })

                    if (split == 'train' and i + 1 >= Config.train_size) or (split == 'dev' and i + 1 >= Config.dev_size):
                        break
                json.dump(self.datas, open(Config.input_cache_dir + split + '.json', 'w'))
        else:
            self.datas = json.load(open(Config.input_cache_dir + split + '.json', 'r'))

    def tokenize_retriever(self, src_text):
        tok_text = [self.retriever_tokenizer.encode(sentence) for sentence in src_text]
        input_ids, attention_masks, cls_ids = list(), list(), list()
        idx_offset, sentence_id, chunk_id = 0, 0, 0
        while sentence_id < len(tok_text) and chunk_id < Config.max_chunks:
            input_id = []
            while sentence_id < len(tok_text):
                tok_sentence = tok_text[sentence_id]
                if len(input_id) + len(tok_sentence) > Config.max_retrieval_len:
                    if len(input_id) == 0:
                        tok_sentence = tok_sentence[:Config.max_retrieval_len - len(input_id)]
                    else:
                        break
                input_id.extend(tok_sentence)
                cls_ids.append(len(input_id) - 1 + idx_offset)
                sentence_id += 1
            attention_mask = [1] * len(input_id)

            # Padding
            num_pad = Config.max_retrieval_len - len(input_id)
            input_id.extend([self.retriever_tokenizer.pad_token_id] * num_pad)
            attention_mask.extend([0] * num_pad)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            idx_offset += Config.max_retrieval_len
            chunk_id += 1

        return input_ids, attention_masks, cls_ids

    def tokenize_generator(self, src_text, summary):
        context_input_ids = list()
        # input_ids
        for sentence in src_text:
            input_ids = [1] + self.generator_tokenizer.encode(sentence.replace(' ', '')) + [self.generator_tokenizer.eod_id]
            if len(input_ids) >= Config.max_source_len:
                input_ids = input_ids[:Config.max_source_len - 1]
            # Padding
            input_ids = input_ids + [1]
            num_pad = Config.max_source_len - len(input_ids)
            context_input_ids.append([self.generator_tokenizer.pad_id] * num_pad + input_ids)

        # labels
        labels = self.generator_tokenizer.encode(summary.replace(' ', ''))  + [self.generator_tokenizer.eod_id]
        if len(labels) > Config.max_target_len:
                labels = labels[:Config.max_target_len]
        num_pad = Config.max_target_len - len(labels)
        labels.extend([self.generator_tokenizer.pad_id] * num_pad)

        return context_input_ids, labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]

        retriever_input_ids = torch.LongTensor(data['retriever_input_ids'])
        retriever_attention_masks = torch.LongTensor(data['retriever_attention_masks'])
        cls_ids = torch.LongTensor(data['cls_ids'])
        oracle = torch.LongTensor(data['oracle'])
        context_input_ids = torch.LongTensor(data['context_input_ids'])
        labels = torch.LongTensor(data['labels'])[:Config.max_target_len]

        return retriever_input_ids, retriever_attention_masks, cls_ids, oracle, context_input_ids, labels


class DyleDemoDataset(torch.utils.data.Dataset):
    def __init__(self, str_list: list, retriever_tokenizer, generator_tokenizer) -> None:
        super().__init__()
        self.data = list()
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.nlp = spacy.load("zh_core_web_lg")
        
        for i, text in enumerate(str_list):
            text = text.replace(' ', '')
            text_sents, para_pos = self.cut_sent(text)
            retriever_input_ids, retriever_attention_masks, cls_ids = self.tokenize_retriever(text_sents)
            context_input_ids = self.tokenize_generator(text_sents)
            
            self.data.append({
                'id': i,
                "text_sents": text_sents,
                'para_pos': para_pos,
                'retriever_input_ids': torch.LongTensor(retriever_input_ids),
                'retriever_attention_masks': torch.LongTensor(retriever_attention_masks),
                'cls_ids': torch.LongTensor(cls_ids),
                'context_input_ids': torch.LongTensor(context_input_ids)
            })

    def tokenize_retriever(self, src_text):
        tok_text = [self.retriever_tokenizer.encode(sentence) for sentence in src_text]
        input_ids, attention_masks, cls_ids = list(), list(), list()
        idx_offset, sentence_id, chunk_id = 0, 0, 0
        while sentence_id < len(tok_text) and chunk_id < Config.max_chunks:
            input_id = []
            while sentence_id < len(tok_text):
                tok_sentence = tok_text[sentence_id]
                if len(input_id) + len(tok_sentence) > Config.max_retrieval_len:
                    if len(input_id) == 0:
                        tok_sentence = tok_sentence[:Config.max_retrieval_len - len(input_id)]
                    else:
                        break
                input_id.extend(tok_sentence)
                cls_ids.append(len(input_id) - 1 + idx_offset)
                sentence_id += 1
            attention_mask = [1] * len(input_id)

            # Padding
            num_pad = Config.max_retrieval_len - len(input_id)
            input_id.extend([self.retriever_tokenizer.pad_token_id] * num_pad)
            attention_mask.extend([0] * num_pad)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            idx_offset += Config.max_retrieval_len
            chunk_id += 1

        return input_ids, attention_masks, cls_ids

    def tokenize_generator(self, src_text):
        context_input_ids = list()
        # input_ids
        for sentence in src_text:
            input_ids = [1] + self.generator_tokenizer.encode(sentence.replace(' ', '')) + [self.generator_tokenizer.eod_id]
            if len(input_ids) >= Config.max_source_len:
                input_ids = input_ids[:Config.max_source_len - 1]
            # Padding
            input_ids = input_ids + [1]
            num_pad = Config.max_source_len - len(input_ids)
            context_input_ids.append([self.generator_tokenizer.pad_id] * num_pad + input_ids)

        # labels
        # labels = self.generator_tokenizer.encode(summary.replace(' ', ''))  + [self.generator_tokenizer.eod_id]
        # if len(labels) > Config.max_target_len:
        #         labels = labels[:Config.max_target_len]
        # num_pad = Config.max_target_len - len(labels)
        # labels.extend([self.generator_tokenizer.pad_id] * num_pad)

        return context_input_ids

    def cut_sent(self, text: str):
        # para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
        # para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
        # para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
        # para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # para = para.rstrip()
        # return para.split("\n")
        paras = [line.strip() for line in text.split('\n')]
        paras = [line for line in paras if len(line) > 0]
        para_pos = []
        sentences = []
        for para in paras:
            doc = self.nlp(para)
            sentences.extend([sent.text for sent in doc.sents])
            para_pos.append(len(sentences) - 1)

        return sentences, para_pos
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_i = self.data[index]
        return data_i['id'], data_i["text_sents"], data_i["para_pos"], data_i["retriever_input_ids"], data_i["retriever_attention_masks"], \
                data_i["cls_ids"], data_i["context_input_ids"]

def dyle_collate_fn(data: List[Dict]):
    keys = data[0].keys()
    output = dict()
    for key in keys:
        output[key] = [item[key] for item in data]
    not_tenser = ["id", "text_sents", "para_pos"]
    for key in output:
        if not (key in not_tenser):
            output[key] = torch.stack(output[key])
    return output['id'], output["text_sents"], output["para_pos"], output["retriever_input_ids"], \
            output["retriever_attention_masks"],  output["cls_ids"], output["context_input_ids"]
    