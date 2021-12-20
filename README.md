## Prerequisites
```
conda create -n stower python=3.7
conda activate stower
```
* [transformers](https://github.com/huggingface/transformers) >= 4.4.2
  ```
  pip install transformers
  ```
* Install our fairseq repo
  ```
  cd VT-STOWER
  pip install --editable ./
  ```
* [hydra](https://github.com/facebookresearch/hydra) = 1.0.3
  ```
  pip install hydra-core==1.0.3
  ```
* boto3
  ```
  pip install boto3
  ```
## Our outputs
For simplity, we directly put our outputs under the folder `outputs`.
  
## Data Preprocessing
Dataset is stored at `preprocess` folder. To preprocess dataset, run the command:
 ```
 cd preprocess
 bash prepare.sh
 ```
After preprocessing, three databins will be generated: 
 * `yelp-databin`: yelp dataset for sentiment transfer;
 * `gyafc-databin`: gyafc dataset for formality transfer;
 * `en-hi-databin`: Hindi-Hinglish dataset for code-switching transfer.

## Training
Training scripts are stored in the `train` folder
```
cd train
bash train-{yelp,gyafc,en-hi}.sh
```
Training bash files include stage I, classifier, and stage II training process.

## Style Transfer
To transfer style, please run the commands under `generate` folder:
```
cd generate
bash generate-{yelp,gyafc,en-hi}.sh
```
Outputs are named as `*.out`


