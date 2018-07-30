#!/usr/bin/env bash

# Run this script from the root path of the nmtlab resposity.

echo "Downloading IWSLT15 vi-en data ..."
DATA_DIR="private/iwslt15_vien"
mkdir -p $DATA_DIR
curl -o "$DATA_DIR/iwslt15_train.vi" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi"
curl -o "$DATA_DIR/iwslt15_train.en" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en"
curl -o "$DATA_DIR/iwslt15_tst2013.vi" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi"
curl -o "$DATA_DIR/iwslt15_tst2013.en" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en"
curl -o "$DATA_DIR/iwslt15_tst2012.vi" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi"
curl -o "$DATA_DIR/iwslt15_tst2012.en" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en"


