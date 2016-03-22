#!/usr/bin/env bash
while read SYMBOL
do
    ./hmm_train.py "$SYMBOL"
done < symbols
