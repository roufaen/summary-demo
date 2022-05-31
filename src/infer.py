#coding:utf-8

import time
import random
import torch
import bmtrain as bmp
import numpy as np
import os
import json
from model_center.model import CPM1Config,CPM1 
from model_center.tokenizer import CPM1Tokenizer 

from model_center import get_args
from .generation import generate

from .infer_dataset import INFER_DATASET

from .config import Config

class Summarizer:

    def __init__(self):
        bmp.init_distributed(seed=1234, loss_scale_factor=2, loss_scale_steps=1024)
        self.tokenizer = CPM1Tokenizer(Config.model_path + 'vocab.txt')
        self.config = CPM1Config.from_json_file(Config.model_path + 'config.json')
        self.config.vocabsize = self.tokenizer.vocab_size
        self.model = CPM1(self.config)
        bmp.load(self.model, Config.model_path + 'pytorch_model.pt')

# def get_model(args, vocab_size):
#     config = 
#     config.vocab_size = vocab_size
#     print ("vocab size:%d"%(vocab_size))
# 
#     model = CPM1(config)
#     # if args.load != None:
#     bmp.load(model, args.load)
#     # else:
#     #     bmp.init_parameters(model)
#     return model
# 
# def setup_model(args):
#     tokenizer = get_tokenizer(args)
#     model = get_model(args, tokenizer.vocab_size)
#     # bmp.synchronize()
#     # bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
#     # bmp.synchronize()
#     return tokenizer, model
# 
# def initialize():
#     # get arguments
#     args = get_args()
#     # init bmp 
#     # bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
#     # init save folder
#     if args.save != None:
#         os.makedirs(args.save, exist_ok=True)
#     return args

    def get_summary(self, src: str):
        # args = initialize()
        # tokenizer, model = setup_model(args)

        # fout = open("{}.{}".format(args.output_file, bmp.rank()), "w", encoding="utf-8")

        # dataset = INFER_DATASET[args.dataset_name](args.input_file, args.max_length)
        # total_lines = len(dataset)
        # step = (total_lines + bmp.world_size() -1) // bmp.world_size()
        # for idx in range(step):
            # print(bmp.world_size())
        # data_idx = step * bmp.rank() + idx

        # bmp.print_rank(idx)

        # text, golden_summary = dataset[data_idx]
        src = '“' + src + '”的摘要是:'

        # target_span_len = Config.span_length
        # 每个instance指定不同的target span长度
        # target_span_len = int(len(instance['source'][0])*0.4*0.7)

        # TODO: support multi-GPUs for varied target span length
        # if target_span_len != args.span_length:
        #     assert bmp.world_size() == 1, "Using multiple GPUs for varied target span length has not been supported!"

        # 指定最短生成长度
        # min_len = min(target_span_len-1, int(len(instance['source'][0])*0.4*0.7))
        # min_len = 2 # 确保生成内容不为空

        predict_sentence = ""

        for it in generate(self.model, self.tokenizer, src, Config.span_length, beam=Config.beam_size,
                            temperature = Config.temperature, top_k = Config.top_k, top_p = Config.top_p,
                            no_repeat_ngram_size = Config.no_repeat_ngram_size, repetition_penalty = Config.repetition_penalty, 
                            random_sample=Config.random_sample, min_len=2):
            if it == '<eod>':
                break

            predict_sentence += it
            # fout.write(it)
            # fout.flush()

        # result_dict = {
        #     "summary": predict_sentence,
        #     "text": text
        # }


        # if data_idx >= total_lines:
        #     continue

        # fout.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        # fout.flush()

        # fout.close()
        return predict_sentence

# if __name__ == "__main__":
#     main()
