#!/usr/bin/env bash
FILES=$(ls weights/aug_dataset_50/yolact_basic_*.pth)
for FILE in $FILES
do
echo $FILE
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model=$FILE --score_threshold=0.15
done
