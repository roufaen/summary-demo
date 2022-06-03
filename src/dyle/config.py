class Config:
    # learning rate
    max_grad_norm = 1.0
    retriever_start_lr = 5e-5
    generator_start_lr = 5e-5
    max_decay_num = 3
    no_improvement_decay = 2

    # model
    max_retrieval_len = 512
    max_chunks = 50
    top_k = 6
    max_source_len = 100
    max_target_len = 100
    retriever_alpha = 1
    generation_alpha = 0.5
    consistency_alpha = 1

    # training
    # train_size = 10000
    # dev_size = 300
    train_size = 1e9
    dev_size = 1e9
    gradient_accumulation_steps = 8
    save_steps = 20000
    train_log_steps = 10000
    validation_log_steps = 1000

    # inference
    infer_beam_size = 5
    infer_temperature = 1
    infer_length_penalty = 1
    infer_top_k = 0
    infer_top_p = 0
    infer_no_repeat_ngram_size = 0
    infer_repetition_penalty = 1
    infer_random_sample = False

    # directory
    base_dir = '/data2/private/huangshuhong/'
    input_cache_dir = base_dir + 'DYLE_repro/input_cache/'
    output_dir = base_dir + 'DYLE_repro/outputs/'
    save_model_dir = base_dir + "model/checkpoints/"
    model_path = base_dir + 'model/cpm1-small/'
    data_path = base_dir + 'datasets/CNewSum/'
    name = ""
