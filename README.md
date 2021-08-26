# Dependency
* python 3.6/3.7
* fairseq, which will also automatically install pytorch
* [transformers](https://huggingface.co/transformers/)
* tensorflow 1.4
* [fasttext](https://github.com/facebookresearch/fastText) (Install with the following evaluation step)
* mosesdecoder
* boto3 >= 1.9.107
* requests >= 2.21.0
* regex >= 2019.2.21

# External repo (they are folders out of fairseq repo)
* VACS
* fasttext
* transformer-drg-style-transfer

All data, models and commands can be found under `vae-fairseq`. I am going to introduce how to train and evaluate models step by step.

# Data Preprocessing
Usually, the dataset is split into `train.0`, `train.1`, `dev.0`, `dev.1`, `test.0`, `test.1` (assume they are in the folder named `raw`), which respectively represent the train, dev and test files in class `0` and `1`. Note that these files are already tokenized and treated as target files. For instance, those files in Yelp dataset are already been tokenized and we should decode the sentences in a seq2seq model based on the original tokenization during generation. The input data should follow the steps that 1) it is tokenized by the corresponding pre-trained language model, 2) a symbol at the end of the sentence should identify the class of the sentence, 3) input data is mixed data with both classes. We first create a folder `preprocessed` used for storing proccessed files.
## Tokenization and Adding Special Symbols
To finish step 1) and 2), we run:
```
python transform_tokenizer.py --input $INPUT_FILE --output $OUTPUT_FILE --pretrained_model $PRE_TRAINED --suffix $CLASS
```
It will tokenize the input data and store it to the target position, and it will also add a speical symbol at the end of the sentence. Spefically, `<s>`(BOS) for class `0` and `</s>`(EOS) for class 1. An example of tokenizing the input file `train.0` with the pre-trained model `RoBERTa` is shown as follows.
```
python transform_tokenizer.py --input raw/train.0 --output preprocessed/train.0 --pretrained_model roberta-base --suffix 0
```

## Mixing Dataset
To finish step 3), we run following commands:
```
cd preprocessed
cat train.0 train.1 > train.all.0   # Combine tgt files of two classes into 1 file
cat ../raw/train.0 ../raw/train.1 > train.all.1 # Combine src files of two classes into 1 file
paste -d '@@@' train.all.0 /dev/null /dev/null ../raw/train.all.1 | shuf > train.all  # Concatenate src and tgt sentences with "@@@", and shuf them
cat train.all | awk -F'@@@' '{print $1}' > train.0 # Get the first part of sentences as new src sentences
cat train.all | awk -F'@@@' '{print $2}' > train.1 # Get the second part of sentences as new tgt sentences
rm train.all*
```
Note that `0` and `1` here simply represent src and tgt files, which are the same content but different tokenization.  We also do the same process to `dev` files.
If we want to transfer from class `1` to `0`, we should 
```
mv test.1 test.0   # the input should be the class 1 file.
cp ../raw/test.0 test.1
```
## Get the Vocabulary of Pre-trained Language Model
We also have to get the vocabulary to convert tokens into indices during preprocessing. To get the vocab of a pretrained model:
```
python get_vocab.py --tokenizer robert-base --output ./preprocessed/src_vocab.txt
```
## Get the Vocabulary of Raw files
To get the vocab from a text file, we run `python`, and then:
```
from dictionary import Dictionary
d = Dictionary(src=False)
with open("tgt_vocab.txt", "w") as f:
    for i in range(len(d)):
        f.writelines([d.id2word[i], "\n"])
```

## Fairseq Preprocessing Command
```
TEXT=./preprocessed/
fairseq-preprocess --source-lang 0 --target-lang 1  --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/valid --destdir ${TEXT}/databin --srcdict $TEXT/src_vocab.txt --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25
```

## Preprocessed files:
Preprocessing has many steps, but we also have preprocessed files for sentiment transfer and code-switching transfer, which respectively located at `vae-fairseq/data/{all-yelp, all-en-hi}`. Raw files are located at `vae-fairseq/data/{raw-yelp, raw-cs}`

# Training

## Stage 1 Training: Training style embeddings and VAE
For code-switching data whose size is 7K:
```
TEXT=./preprocessed/
SAVE_DIR=./models/cs/

CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train ${TEXT}cs-databin/ --arch cs-vae --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0  \
--lr 0.0005 --lr-scheduler inverse_sqrt  --dropout 0.1 --warmup-updates 1000 --warmup-init-lr 1e-07 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 1024 --update-freq 1 --attention-dropout 0.1 --activation-dropout 0.1 --max-update 100000 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 --max-epoch 201  \
--save-interval 3  --alpha 0.5 --no-epoch-checkpoints  --weight_c 1  --pretrained_model xlm-roberta-base --vae_encoder_layers 1 --vae_type base 
```
Note that we use a smaller model architecture `cs-vae` with 2 transformer layers, 2 attention heads and 256 forward dimension

For sentiment data whose size is 440K, we use a lagrer model architecture `vae` with 2 transoformer layers, 4 heads and 1024 forward dimension.
```
CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train ${TEXT}neg-databin/ --arch vae --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0  \
--lr 0.0005 --lr-scheduler inverse_sqrt  --dropout 0.1 --warmup-updates 1000 --warmup-init-lr 1e-07 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 8192 --update-freq 1 --attention-dropout 0.1 --activation-dropout 0.1 --max-update 100000 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 --max-epoch 201  \
--save-interval 3  --alpha 1 --no-epoch-checkpoints  --weight_c 1  --pretrained_model roberta-base --vae_encoder_layers 1 --vae_type base
```

Our stage 1 trained models are located at `vae-fairseq/models/{yelp,cs}/vae+style_embed`

## Stage 2 Training: Training with Masked Pivot Words
Pre-train a classifier first (take RoBerta as an example):
### Prepare data for classification training:
```
python BERT_Classification_Training_data_preparation.py ## remember to change the data path in this py file
```
### Classification Training
```
export BERT_DATA_DIR=raw # raw data path
export BERT_MODEL_DIR=models/classification/  #Path to save the classifier trained model
CUDA_VISBLE_DEVICES=0 python run_classifier.py \
--data_dir=$BERT_DATA_DIR \
--bert_model=roberta-base \
--output_dir=$BERT_MODEL_DIR \
--max_seq_length=128 \
--do_train \
--train_batch_size=128 \
--num_train_epochs=10 
```
### Classification Evaluation
```
CUDA_VISBLE_DEVICES=0 python run_classifier.py \
--data_dir=$BERT_DATA_DIR \
--bert_model=roberta-base \
--output_dir=$BERT_MODEL_DIR \
--max_seq_length=128 \
--do_eval \
--train_batch_size=128 \
--num_train_epochs=10 
```

Our classifiers are located at `transformer-drg-style-transfer/model/{yelp, cs}/pytorch_model.bin`

### Start training VAE and frozen style embeddings with masked pivot words:
For code-switching:
```
TEXT=./preprocessed/
SAVE_DIR=./models/cs/

CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train ${TEXT}cs-databin/ --arch cs-vae --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0  \
--lr 0.0005 --lr-scheduler inverse_sqrt  --dropout 0.1 --warmup-updates 1000 --warmup-init-lr 1e-07 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 1024 --update-freq 1 --attention-dropout 0.1 --activation-dropout 0.1 --max-update 100000 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 --max-epoch 401  \
--save-interval 3  --alpha 0.5 --no-epoch-checkpoints  --weight_c 1  --pretrained_model xlm-roberta-base --vae_encoder_layers 1 --vae_type base \
--warmup_from_nmt --warmup_nmt_file ./models/cs/checkpoint_best.pt --reset-lr-scheduler --reset-optimizer --stage 1  --score_maker models/classification/pytorch_model.bin
```
where `--warmup_from_nmt` indicates the location of model in stage 1, `--score_maker` indicates the path of classification model, and `--stage` indicate the the stage of training (`0` means stage 1 and `1` means stage 2. Default number is `0`)

For sentiment transfer:
```
CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train ${TEXT}neg-databin/ --arch vae --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0  \
--lr 0.0005 --lr-scheduler inverse_sqrt  --dropout 0.1 --warmup-updates 1000 --warmup-init-lr 1e-07 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 8192 --update-freq 1 --attention-dropout 0.1 --activation-dropout 0.1 --max-update 100000 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 --max-epoch 401  \
--save-interval 3  --alpha 1 --no-epoch-checkpoints  --weight_c 1  --pretrained_model roberta-base --vae_encoder_layers 1 --vae_type base \
--warmup_from_nmt --warmup_nmt_file ./models/yelp/checkpoint_best.pt --reset-lr-scheduler --reset-optimizer --stage 1  --score_maker models/classification/pytorch_model.bin

```
Our stage 2 models are located at `vae-fairseq/models/{yelp,cs}/vae+style_embed+mask+score/`

# Style Transfer (Generation)
To transfer from one style to another style, we run the fairseq command (remember the input file is `test.0` in the preprocessed folder):
```
MODELPATH=./models/yelp/
PRE=roberta-base
PRE_SRC=roberta-base
## transfer from 1 -> 0
STPATH=${DATAPATH}neg-databin/

CUDA_VISIBLE_DEVICES=6 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC}  --style_weight 1.55 \
--beam 10 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.1.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out

cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > ${STPATH}/generate.extract

```
The output is the file `generate.extract`.

# Evaluation
We have three main metrics to evaluate the performance of style transfer, i.e., accuracy of a binary classifier on transferred texts, perplexity and BLEU scores.
## Binary Classifier
Following previous methods, we use [FastText classifer](https://fasttext.cc/docs/en/supervised-tutorial.html).
### Install fastext classifier:
```
wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
unzip v0.9.2.zip
cd fastText-0.9.2
make
pip install .
```
if it fails, please try to install it base on [this repo](https://github.com/facebookresearch/fastText)  


### Preprocess the data to accomandate fasttext classfier input format
Every line should have a label `__label__0` or `__label__1` to indicate the class of the sentence. Before we train the classifier, we should preprocess the files in `raw` folder:
```
cd fasttext-0.9.2
mkdir data
awk '$0="__label__0 "$0' ../raw/train.0 > data/train_0.txt
awk '$0="__label__1 "$0' ../raw/train.1 > data/train_1.txt
cat data/train_0.txt data/train_1.txt | shuf > data/train.txt
```
We can conduct the same thing for `test.{0,1}` files
### Train the classifier:
we run in `python`:
```
import fasttext
model = fasttext.train_supervised(input="train.txt", lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
model.save_model("models/model.pt")
```
### Evaluate the classifier:
Use command line:
```
fasttext test models/model.pt test.0
```
The results will show the precision and recall. Because we only have one label for the test file, the accuracy equals to the recall.


Our classifiers are located at `vae-fairseq/models/classifier/{yelp, cs}.pt`

## Perplexity
### PPL for pure English transfer
For pure English tasks like sentiment transfer, PPL calculation is very easy by the GPT2 model. For instance, to calculate the ppl of the output file `generate.extract`:
```
CUDA_VISIBLE_DEVICES=0 python cal_ppl.py --input generate.extract
```
### PPL for code-switching transfer
It would be much more complicated when it comes to code-switching transfer. We will first train customized [LSTM language model](https://github.com/bidishasamantakgp/VACS/tree/master/language_model)
```
cd VACS/language_model 
git checkout train
```
This repo is old and a little hardcoded, where the name of train, dev, and test files have to be `train.txt`, `valid.txt`, `test.txt` and under the folder `csdata`, where they are derived from
```
cat ../../raw/train.0 ../../raw/train.1 > csdata/train.txt
cat ../../raw/dev.0 ../../raw/dev.1 > csdata/valid.txt
cat ../../raw/test.0 ../../raw/test.1 > csdata/test.txt
```

For training:
```
python train.py
```
The model will be stored in the folder `cv-cs`.

Evaluation for the `test.txt`:
```
python evaluate.py --load_model cv-cs/epoch016_4.5584.model --data_dir csdata
```

However, if we want to evaluate the external test file, it will raise vocab incosistence error because this repo is hardcoded and vocab is define by train, dev and test files. If we change the test file, we vocab will be different from the one in the model. We therefore modify the repo to allow to accept external test files. e.g., `generate.extract` file generated by fairseq:
```
git checkout eval
mv csdata/test.txt csdata/test2.txt
cp generate.extract csdata/test.txt
python evaluate.py --load_model cv-cs/epoch016_4.5584.model --data_dir csdata
```
It will show the ppl of the input file `generate.extract`

Our language model is located at `VACS/language_model/cv-cs/epoch016_4.5584.model`

### BLEU scores
The bleu scores are calculated by `mosedecoder`: 
```
git clone https://github.com/moses-smt/mosesdecoder.git
```
An example of BLEU score calculation between the original file (`test.1`) and generated file (`generate.extract`) is:
```
cat generate.extract | modesdecoder/scripts/generic/multi-bleu.perl raw/test.1
```

## Pipline of Evaluation
For convenience, we also have a evaluation bash file for the whole process of 3 evaluation metrics. `eval.sh` for sentiment transfer and `eval_cs.sh` for code-switching transfer. Please remember change the path of files when you use the bash file.




