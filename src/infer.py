#coding:utf-8

from re import L
import torch
import spacy
from .config import InferConfig, SegmentConfig
from .textseg.demo import SegBertDemo
from model_center.model import CPM1Config,CPM1 
from model_center.tokenizer import CPM1Tokenizer 
import torch.distributed as dist

from model_center import get_args
from .generation import generate

from .infer_dataset import BatchInferDataset, DemoSumInferDataset
from .dyle.infer import DyleInfer


class Summarizer:
    def __init__(self, config: InferConfig):
        self.tokenizer = CPM1Tokenizer(config.model_path + "/vocab.txt")
        self.model_config = CPM1Config.from_json_file(config.model_path + "/config.json")
        self.model_config.vocab_size = self.tokenizer.vocab_size
        print ("vocab size:%d"%(self.model_config.vocab_size))
        self.model = CPM1(self.model_config).cuda()
        self.model.load_state_dict(
            torch.load(config.load_path),
            strict=True
        )
        self.config = config
        torch.cuda.synchronize()
    
    def summarize(self, str_list: list):
        dataset = DemoSumInferDataset(str_list, self.tokenizer, self.config.max_length)
        step = len(str_list)
        batch_num = (step + self.config.batch_size - 1) // self.config.batch_size
        batch_dataset = BatchInferDataset(dataset, self.tokenizer, 
                                          self.config.span_length, 
                                          self.config.batch_size, batch_num)
        min_len = 2 # 确保生成内容不为空

        def work(input_dict):
            result = generate(self.model, self.tokenizer, input_dict, beam=self.config.beam_size,
                            temperature = self.config.temperature, length_penalty=self.config.length_penalty, 
                            top_k = self.config.top_k, top_p = self.config.top_p,
                            no_repeat_ngram_size = self.config.no_repeat_ngram_size, repetition_penalty = self.config.repetition_penalty, 
                            random_sample=self.config.random_sample, min_len=min_len)
            output = []
            for idx, sent in enumerate(result):
                output.append({
                    "sentence": sent,
                    "id": input_dict['ids'][idx]
                })
            return output
        
        all_output = []
        for input_dict in batch_dataset:
            if input_dict["valid"]:
                output = work(input_dict)
                all_output.extend(output)
        
        all_output = sorted(all_output, key=lambda x: x["id"])
        
        all_summaries = [item["sentence"] for item in all_output]
        return all_summaries


class Segmentator:
    def __init__(self, config: SegmentConfig):
        self.config = config
        self.nlp = spacy.load("zh_core_web_lg")
        self.demo = SegBertDemo(config.model_path, config.model_config_path,
                                batch_size=config.batch_size, device=config.device)

    def cut_sent(self, text: str):
        paras = [line.strip() for line in text.split('\n')]
        paras = [line for line in paras if len(line) > 0]
        para_pos = []
        sentences = []
        for para in paras:
            doc = self.nlp(para)
            sentences.extend([sent.text for sent in doc.sents])
            para_pos.append(len(sentences) - 1)

        return sentences, para_pos

    def get_segments(self, input: str):
        """
        1. 先把文章按照\n分段，记录其位置，然后用spacy分句得到句子列表
        2. 将该列表用text segmentation处理后得到分段信息
        3. 恢复句子中的\n
        4. 根据分段信息将句子重组为段落，返回的是一个没有丢失换行信息的段落的列表
        """
        text_sents, para_pos = self.cut_sent(input)
        split_pos = self.demo.get_segmentation(text_sents, min_length=self.config.min_length,
                                          max_length=self.config.max_length,
                                          prob_threshold=self.config.device)
        for pos in para_pos:
            text_sents[pos] = text_sents[pos] + "\n"
        output = []
        split_pos = [0] + split_pos + [len(text_sents)]
        for i in range(len(split_pos) - 1):
            output.append("".join(text_sents[split_pos[i]: split_pos[i + 1]]))
        
        return output

# def get_tokenizer(args):
#     tokenizer = CPM1Tokenizer(args.vocab_file)
#     return tokenizer

# def get_model(args, vocab_size):
#     config = CPM1Config.from_json_file(args.model_config)
#     config.vocab_size = vocab_size
#     print ("vocab size:%d"%(vocab_size))

#     model = CPM1(config).cuda()
#     # if args.load != None:
#     model.load_state_dict(
#         torch.load(args.load),
#         strict = True
#     )
#     torch.cuda.synchronize()
#     return model

# def setup_model(args):
#     tokenizer = get_tokenizer(args)
#     model = get_model(args, tokenizer.vocab_size)
#     return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    # print(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl")

    # init save folder
    # if args.save != None:
    #     os.makedirs(args.save, exist_ok=True)
    return args


# def main():
#     args = initialize()
#     tokenizer, model = setup_model(args)

#     fout = open("{}.{}".format(args.output_file, args.local_rank), "w", encoding="utf-8")

#     dataset = INFER_DATASET[args.dataset_name](args.input_file, args.max_length)
#     total_lines = dataset.total_length
#     step = (total_lines + dist.get_world_size() -1) // dist.get_world_size()
#     dataset.read_dataset(step * args.local_rank, step * (args.local_rank + 1), tokenizer)
#     batch_num = (step + args.batch_size - 1) // args.batch_size
#     batch_dataset = BatchInferDataset(dataset, tokenizer, args.span_length, args.batch_size, batch_num)
#     min_len = 2 # 确保生成内容不为空
#     def work(input_dict):
#         result = generate(model, tokenizer, input_dict, beam=args.beam_size,
#                         temperature = args.temperature, length_penalty=args.length_penalty, top_k = args.top_k, top_p = args.top_p,
#                         no_repeat_ngram_size = args.no_repeat_ngram_size, repetition_penalty = args.repetition_penalty, 
#                         random_sample=args.random_sample, min_len=min_len)
#         for idx, sent in enumerate(result):
#             fout.write(sent + '\t' + str(input_dict['ids'][idx]) + '\n')
#             fout.flush()

#     if args.local_rank == 0:
#         for input_dict in tqdm(batch_dataset):
#             if input_dict['valid']:
#                 work(input_dict)
#     else:
#         for input_dict in batch_dataset:
#             if input_dict['valid']:
#                 work(input_dict)
        
#     fout.close()

# if __name__ == "__main__":
#     main()
