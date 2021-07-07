#!/usr/bin/env bash

python ./src/data/processing/data_preparation.py \
--data_path=./data/processed/chb-mit/features/sec_30 \
--store_final_dir_path=./data/processed/chb-mit/final/sec_30 \
--patient=16 \
--preictal_duration=30 \
--discard_data_duration=60 \
--features_names univariate max_correlation
