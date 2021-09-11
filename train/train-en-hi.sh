## Stage I Training
TEXT=../preprocess/en-hi-databin/
SAVE_DIR=../models/en-hi/
TASK=cs

CUDA_VISIBLE_DEVICES=0 fairseq-train ${TEXT}/databin-01/ --arch cs-vae --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0  \
--lr 0.0005 --lr-scheduler inverse_sqrt  --dropout 0.1 --warmup-updates 1000 --warmup-init-lr 1e-07 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 1024 --update-freq 3 --attention-dropout 0.1 --activation-dropout 0.1 --max-update 100000 --save-dir ${SAVE_DIR}stage1  --encoder-embed-dim 768 --decoder-embed-dim 768 --max-epoch 200  \
--save-interval 1  --alpha 1 --no-epoch-checkpoints  --weight_c 1  --pretrained_model xlm-roberta-base --vae_encoder_layers 1 --vae_type base --transfer_task ${TASK}

## Classifier Training
CUDA_VISBLE_DEVICES=0 python run_classifier.py \
--data_dir=${TEXT} \
--bert_model=xlm-roberta-base \
--output_dir=${SAVE_DIR}classifier \
--max_seq_length=128 \
--do_train \
--train_batch_size=128 \
--num_train_epochs=10

## Stage II Training
CUDA_VISIBLE_DEVICES=0 fairseq-train ${TEXT}/databin-01/ --arch cs-vae --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0  \
--lr 0.0005 --lr-scheduler inverse_sqrt  --dropout 0.1 --warmup-updates 1000 --warmup-init-lr 1e-07 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 1024 --update-freq 3 --attention-dropout 0.1 --activation-dropout 0.1 --save-dir ${SAVE_DIR}stage1  --encoder-embed-dim 768 --decoder-embed-dim 768 --max-epoch 600  \
--save-interval 1  --alpha 1 --no-epoch-checkpoints  --weight_c 1  --pretrained_model xlm-roberta-base --vae_encoder_layers 1 --vae_type base --transfer_task ${TASK}
--warmup_from_nmt --warmup_nmt_file ${SAVE_DIR}stage1/checkpoint_best.pt --reset-lr-scheduler --reset-optimizer --stage 1 --score_maker ${SAVE_DIR}classifier/pytorch_model.bin
