DATAPATH=../preprocess/en-hi-databin/
MODELPATH=../models/en-hi/stage2/
PRE=xlm-roberta-base
PRE_SRC=xlm-roberta-base

## from pure->code-switching (1->0)
STPATH=${DATAPATH}databin-10/

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC}  --style_weight 0.75 \
--beam 10 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.1.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > en-hi.10.out
