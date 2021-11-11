#!/bin/bash
shuf data/pos_processed_gnps.txt > data/pos_processed_gnps_shuffled.txt
tail -n +137001 data/pos_processed_gnps_shuffled.tsv > data/pos_processed_gnps_shuffled_validation.tsv
head -n 137000 data/pos_processed_gnps_shuffled.tsv > data/pos_processed_gnps_shuffled_train.tsv
