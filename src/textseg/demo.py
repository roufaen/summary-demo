import spacy
import torch

import numpy as np

from typing import List, Tuple
from transformers import AutoModel, BertTokenizer, BertConfig
from transformers.tokenization_utils import PaddingStrategy
from .segbert import CrossSegBert


class SegBertDemo:
    """Cross Segment BERT demo for text segmentation
    """
    
    def __init__(self, model_path, model_config_path, batch_size=256, max_context_length=254, device='cpu'):
        # self.nlp = spacy.load("zh_core_web_lg")
        
        self.model = CrossSegBert(AutoModel.from_config(BertConfig.from_pretrained(model_config_path))).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_config_path)
        self.batch_size = batch_size
        self.max_context_length=254
        self.device = device
        
    def get_segmentation(self, sentences: List[str], min_length=200, max_length=600, prob_threshold=0.5) -> List[int]:
        """Get text segmentation result by length
        """
        sentences, text_id = self._input_processing(sentences)
        prob = self.get_segmentation_prob(text_id)
        sent_length_list = [len(sentence) for sentence in sentences]
        sorted_prob_list = sorted(list(zip(prob, list(range(1, len(prob) + 1)))), reverse=True)
        section_length_list = [sum(sent_length_list)]
        start_list = [0]
        end_list = [len(sentences)]
        for prob_and_pos in sorted_prob_list:
            if max(section_length_list) > max_length or prob_and_pos[0] >= prob_threshold:
                pos = prob_and_pos[1]
                for i in range(len(end_list)):
                    if end_list[i] > pos:
                        if sum(sent_length_list[start_list[i]:pos]) < min_length or sum(sent_length_list[pos:end_list[i]]) < min_length:
                            break
                        start_list.insert(i+1, pos)
                        end_list.insert(i, pos)
                        section_length_list.pop(i)
                        section_length_list.insert(i, sum(sent_length_list[start_list[i]:end_list[i]]))
                        section_length_list.insert(i+1, sum(sent_length_list[start_list[i+1]:end_list[i+1]]))
                        break
            else:
                break
        split_pos = end_list[:-1]
        return split_pos
    
    def get_segmentation_by_prob(self, text: str, prob_threshold=0.5) -> List[int]:
        """Get text segmentation result by prob threshold
        
        Args:
            text: raw text for segmentation

        Returns:
            A list of text
        """
        sentences, text_id = self._input_processing(text)
        prob = self.get_segmentation_prob(text_id)
        split_pos = np.where(np.array(prob) >= prob_threshold)[0] + 1
        return split_pos
        
    def get_segmentation_prob(self, text_id: List[List[int]]) -> List[float]:
        context_list = [self._get_context(text_id, mid_sent_idx) for mid_sent_idx in range(1, len(text_id))]
        batch_num = (len(context_list) + self.batch_size - 1) // self.batch_size
        prob = []
        for batch_idx in range(batch_num):
            input_context = context_list[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            prob.extend(self._call_model(input_context))
        return prob
    
    def _input_processing(self, sentences: List[str]) -> Tuple[List[str], List[List[int]]]:
        text_id = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)) for sent in sentences]
        return sentences, text_id
    
    def _get_context(self, text_id: List[List[int]], mid_sent_idx: int) -> Tuple[List[int], List[int]]:
        left_context = []
        left_idx = mid_sent_idx - 1
        while 0 <= left_idx and len(text_id[left_idx]) + len(left_context) <= self.max_context_length:
            left_context = text_id[left_idx] + left_context
            left_idx -= 1
        if 0 <= left_idx and self.max_context_length > len(left_context):
            left_context = text_id[left_idx][len(left_context) - self.max_context_length:] + left_context
            
        right_context = []
        right_idx = mid_sent_idx
        while right_idx < len(text_id) and len(text_id[right_idx]) + len(right_context) <= self.max_context_length:
            right_context += text_id[right_idx] 
            right_idx += 1
        if right_idx < len(text_id) and self.max_context_length > len(right_context):
            right_context += text_id[right_idx][:self.max_context_length - len(right_context)]
            
        return [left_context, right_context]
            
    def _call_model(self, input_context: List[Tuple[List[int], List[int]]]) -> List[int]:
        model_input = self.tokenizer._batch_prepare_for_model(input_context, padding_strategy=PaddingStrategy.MAX_LENGTH, max_length=512, return_tensors='pt')
        model_input = {key: value.to(self.device) for key, value in model_input.items()}
        logits = self.model(**model_input)
        prob = torch.softmax(logits, dim=-1)
        return list(prob[:, 1].detach().cpu().numpy())
    
    def _post_processing(self, sentences: List[str], split_pos: List[int]) -> List[str]:
        text_list = []
        start = 0
        for end in split_pos:
            text_list.append(''.join(sentences[start:end]))
            start = end
        text_list.append(''.join(sentences[start:]))
        return text_list
