DATAPATH=../preprocess/yelp-databin/
MODELPATH=../models/yelp/stage2/
PRE=roberta-base
PRE_SRC=roberta-base

## from pos->neg (1->0)
STPATH=${DATAPATH}databin-10/

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC}  --style_weight 2 \
--beam 10 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.1.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > yelp.10.out

## from neg->pos (0->1)
STPATH=${DATAPATH}databin-01/

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC}  --style_weight 2 \
--beam 10 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.1.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > yelp.01.out
