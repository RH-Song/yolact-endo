#!/usr/bin/env bash
FILES=$(ls weights/ciou_aug_dataset_yolactpp/yolact_plus_base_*.pth)
for FILE in $FILES
do
echo $FILE
CUDA_VISIBLE_DEVICES=3 python eval.py --no_bar --trained_model=$FILE --score_threshold=0.15
done
