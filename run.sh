#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13587
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
DATASET="CNewSum"
INPUT_FILE="test.simple.label.jsonl.900"
MODEL_CONFIG_DIR=${CPM_CACHE_PATH}/cpm1-small
EPOCH=$1
CKPT_STEPS=0
LENGTH_PENALTY=1
REPETITION_PENALTY=1
NO_REPEAT_NGRAM_SIZE=0
OUTPUT_FILE=${BASE_PATH}/infer_results/${INPUT_FILE}/${EPOCH}-${CKPT_STEPS}-LP${LENGTH_PENALTY}-RP${REPETITION_PENALTY}-NP${NO_REPEAT_NGRAM_SIZE}.jsonl

if [ ! -d ${BASE_PATH}/infer_results ]; then
    mkdir ${BASE_PATH}/infer_results
fi

if [ ! -d ${BASE_PATH}/infer_results/${INPUT_FILE} ]; then
    mkdir ${BASE_PATH}/infer_results/${INPUT_FILE}
fi

OPTS=""
OPTS+=" --max-length 1024"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --model-config ${MODEL_CONFIG_DIR}/config.json"
OPTS+=" --vocab-file ${MODEL_CONFIG_DIR}/vocab.txt"
OPTS+=" --load ${BASE_PATH}/results/finetune-cpm1-ckpt-${EPOCH}-${CKPT_STEPS}.pt"
OPTS+=" --input-file ${CPM_TRAIN_DATA_PATH}/${DATASET}/${INPUT_FILE}"
OPTS+=" --output-file ${OUTPUT_FILE}"
OPTS+=" --span-length 100"
OPTS+=" --temperature 1"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0"
OPTS+=" --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE}"
OPTS+=" --repetition-penalty ${REPETITION_PENALTY}"
OPTS+=" --length-penalty ${LENGTH_PENALTY}"
OPTS+=" --beam-size 5"
OPTS+=" --batch-size 16"

# CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/code_infer/infer.py ${OPTS}"
# echo ${CMD}
python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} main.py ${OPTS}
# ${CMD} 2>&1 | tee ${BASE_PATH}/infer_results/${INPUT_FILE}/infer-${EPOCH}-${CKPT_STEPS}.log

# cat ${OUTPUT_FILE}.* > ${OUTPUT_FILE}
# rm ${OUTPUT_FILE}.*
