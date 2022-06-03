class InferConfig:
    model_path = '/home/huangshuhong/huangshuhong/model/cpm1-small'
    load_path = '/home/zhaoxinhao/data2/cpm1/experiments/20220505_1_CNewSum/results/finetune-cpm1-ckpt-6-0.pt'

    max_length = 1024

    span_length = 100
    temperature = 1
    top_k = 0
    top_p = 0
    no_repeat_ngram_size = 0
    repetition_penalty = 1
    random_sample = False
    beam_size = 5
    batch_size = 16
    length_penalty = 1


class SegmentConfig:
    model_path = "/data/private/zhaoxinhao/textseg/experiments/20220529_1_multilevel_crossseg/results/huggingface-finetune-cpm1-ckpt-1-0.pt"
    model_config_path = "some path"
    device = "cuda:0"
    batch_size = 16
    min_length = 200
    max_length = 600
    prob_threshold = 0.5
