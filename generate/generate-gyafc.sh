DATAPATH=../preprocess/gyafc-databin/
MODELPATH=../models/gyafc/stage2/
PRE=roberta-base
PRE_SRC=roberta-base

## from informal->formal (1->0)
STPATH=${DATAPATH}databin-10/

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC}  --style_weight 3.1 \
--beam 10 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.1.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > gyafc.10.out

## from formal->informal (0->1)
STPATH=${DATAPATH}databin-01/

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC}  --style_weight 3.1 \
--beam 10 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.1.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > gyafc.01.out
