#!/usr/bin/env bash

python ./src/models/train_model_selection.py \
--final_data_path=./data/processed/chb-mit/final/sec_30 \
--patient=12 \
--network=FC \
--preictal_duration=30 \
--group_segments_form_input=False \
--n_segments_form_input=10 \
--segments_duration=30
