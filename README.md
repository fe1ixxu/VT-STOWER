# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
```bash
pip install --editable ./
```


# Command for training:
```
DEBUG=debug

TEXT=./data/all-yelp/
SAVE_DIR=./models/yelp/${DEBUG}/
nvidia-smi
hostname

CUDA_VISIBLE_DEVICES=4,5,6 fairseq-train ${TEXT}neg-databin/ --arch vae --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0  \
--lr 0.0005 --lr-scheduler inverse_sqrt  --dropout 0.1 --warmup-updates 1000 --warmup-init-lr 1e-07 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 8192 --update-freq 1 --attention-dropout 0.1 --activation-dropout 0.1 --max-update 100000 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 --max-epoch 201  \
--save-interval 3  --alpha 1 --vq_num 2048  --no-epoch-checkpoints  --weight_c 1  --pretrained_model roberta-base --vae_encoder_layers 1 --vae_type base
```

# Command for generation:
```
DATAPATH=./data/all-yelp/
STPATH=${DATAPATH}neg-databin/
MODELPATH=./models/yelp/vae+style_embed/
PRE=roberta-base
PRE_SRC=roberta-base
CUDA_VISIBLE_DEVICES=1 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC}  --batch-size 1 \
--beam 10 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.1.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out

cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > ${STPATH}/generate.extract.sample
```
