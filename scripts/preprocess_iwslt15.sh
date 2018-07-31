#!/usr/bin/env bash

echo "IWSLT15 corpus is already tokenized when downloaded"

# Usually tokenization is required as the first step.

echo "Training truecasing model ..."

RECASER_DIR="third-party/mosesdecoder/scripts/recaser"
BPE_DIR="third-party/subword-nmt/subword_nmt"
DATA_DIR="private/iwslt15_vien"

$RECASER_DIR/train-truecaser.perl --model $DATA_DIR/iwslt15.vi.truecasemodel --corpus $DATA_DIR/iwslt15_train.vi
$RECASER_DIR/train-truecaser.perl --model $DATA_DIR/iwslt15.en.truecasemodel --corpus $DATA_DIR/iwslt15_train.en

echo "Truecasing IWSLT15 corpus ..."

$RECASER_DIR/truecase.perl --model $DATA_DIR/iwslt15.vi.truecasemodel < $DATA_DIR/iwslt15_train.vi > $DATA_DIR/iwslt15_train.truecased.vi
$RECASER_DIR/truecase.perl --model $DATA_DIR/iwslt15.en.truecasemodel < $DATA_DIR/iwslt15_train.en > $DATA_DIR/iwslt15_train.truecased.en
$RECASER_DIR/truecase.perl --model $DATA_DIR/iwslt15.vi.truecasemodel < $DATA_DIR/iwslt15_tst2013.vi > $DATA_DIR/iwslt15_tst2013.truecased.vi
$RECASER_DIR/truecase.perl --model $DATA_DIR/iwslt15.en.truecasemodel < $DATA_DIR/iwslt15_tst2013.en > $DATA_DIR/iwslt15_tst2013.truecased.en

echo "Training BPE model ..."

$BPE_DIR/learn_bpe.py -s 20000 < $DATA_DIR/iwslt15_train.truecased.vi > $DATA_DIR/iwslt15.vi.bpemodel
$BPE_DIR/learn_bpe.py -s 20000 < $DATA_DIR/iwslt15_train.truecased.en > $DATA_DIR/iwslt15.en.bpemodel

echo "Applying BPE model ..."

$BPE_DIR/apply_bpe.py -c $DATA_DIR/iwslt15.vi.bpemodel < $DATA_DIR/iwslt15_train.truecased.vi > $DATA_DIR/iwslt15_train.truecased.bpe20k.vi
$BPE_DIR/apply_bpe.py -c $DATA_DIR/iwslt15.en.bpemodel < $DATA_DIR/iwslt15_train.truecased.en > $DATA_DIR/iwslt15_train.truecased.bpe20k.en
$BPE_DIR/apply_bpe.py -c $DATA_DIR/iwslt15.vi.bpemodel < $DATA_DIR/iwslt15_tst2013.truecased.vi > $DATA_DIR/iwslt15_tst2013.truecased.bpe20k.vi
$BPE_DIR/apply_bpe.py -c $DATA_DIR/iwslt15.en.bpemodel < $DATA_DIR/iwslt15_tst2013.truecased.en > $DATA_DIR/iwslt15_tst2013.truecased.bpe20k.en

echo "Building vocabulary ..."

python ./bin/learn_vocab.py --corpus $DATA_DIR/iwslt15_train.truecased.bpe20k.vi --limit 20000 --vocab $DATA_DIR/iwslt15.truecased.bpe20k.vi.vocab
python ./bin/learn_vocab.py --corpus $DATA_DIR/iwslt15_train.truecased.bpe20k.en --limit 20000 --vocab $DATA_DIR/iwslt15.truecased.bpe20k.en.vocab

echo "Combining bilingual corpus ..."

paste $DATA_DIR/iwslt15_train.truecased.bpe20k.vi $DATA_DIR/iwslt15_train.truecased.bpe20k.en > $DATA_DIR/iwslt15_train.truecased.bpe20k.vien
paste $DATA_DIR/iwslt15_tst2013.truecased.bpe20k.vi $DATA_DIR/iwslt15_tst2013.truecased.bpe20k.en > $DATA_DIR/iwslt15_tst2013.truecased.bpe20k.vien

echo "Done"
