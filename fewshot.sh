#!/bin/bash

for way in 5 10; do
    for shot in 10 20; do
        for i in {0..9}; do
            exp_name="simsiam_transformer_linear_mask_06_concat_${way}w${shot}s"
            echo "Executing command line: fold $i, way $way and shot $shot"
            python main.py --config cfgs/PointSimsiam/fewshot.yaml --finetune_model --ckpts experiments/pretrain/PointSimsiam/simsiam_transformer_linear_mask_06_concat/ckpt-last.pth --exp_name $exp_name --way $way --shot $shot --fold $i
        done
    done
done
