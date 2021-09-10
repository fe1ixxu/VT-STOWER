for TYPE in "yelp" "gyafc" "en-hi" ;
do 
    PRETRAIN=roberta-base
    mkdir ${TYPE}-databin
    mkdir ${TYPE}-temp

    ## Tokenize
    for prefix in "dev" "test" "train" ;
    do
        for suffix in 0 1;
        do
            python transform_tokenize.py --input ${TYPE}/${prefix}.${suffix} --output ${TYPE}-temp/${prefix}.${suffix} --pretrained_model ${PRETRAIN} --suffix ${suffix}
        done
    done

    ## Concatnate both style sentences into one file
    for prefix in "train" "dev" ;
    do 
        cat ${TYPE}-temp/${prefix}.0 ${TYPE}-temp/${prefix}.1 > ${TYPE}-databin/${prefix}.all.0
        cat ${TYPE}/${prefix}.0 ${TYPE}/${prefix}.1 > ${TYPE}-databin/${prefix}.all.1
        paste -d '@@@' ${TYPE}-databin/${prefix}.all.0 /dev/null /dev/null ${TYPE}-databin/${prefix}.all.1 | shuf > ${TYPE}-databin/${prefix}.all
        cat ${TYPE}-databin/${prefix}.all | awk -F'@@@' '{print $1}' > ${TYPE}-databin/${prefix}.0
        cat ${TYPE}-databin/${prefix}.all | awk -F'@@@' '{print $2}' > ${TYPE}-databin/${prefix}.1
        rm ${TYPE}-databin/${prefix}.all*
    done

    cp ${TYPE}-temp/test.0 ${TYPE}-databin/test01.0
    cp ${TYPE}/test.0 ${TYPE}-databin/test01.1
    cp ${TYPE}-temp/test.1 ${TYPE}-databin/test10.0
    cp ${TYPE}/test.1 ${TYPE}-databin/test10.1

    ## Obtain vocab file for fairseq preprocess
    python get_vocab_from_pretrain.py --tokenizer ${PRETRAIN} --output ${TYPE}-databin/src_vocab.txt
    python get_vocab_from_raw.py --input ${TYPE} --output ${TYPE}-databin/tgt_vocab.txt

    ## Fairseq commands
    TEXT=${TYPE}-databin
    fairseq-preprocess --source-lang 0 --target-lang 1  --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test01 \
    --destdir ${TEXT}/databin-01 --srcdict $TEXT/src_vocab.txt --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

    fairseq-preprocess --source-lang 0 --target-lang 1  --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test10 \
    --destdir ${TEXT}/databin-10 --srcdict $TEXT/src_vocab.txt --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

    ## Classifier data preprocessing
    python classifier_data_prepare.py --input ${TYPE} --output ${TYPE}-databin

    rm -rf ${TYPE}-temp
done