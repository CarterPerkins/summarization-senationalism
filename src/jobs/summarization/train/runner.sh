#!/bin/bash

epochs=$1
model=$2
batch_size=$3
lr=$4
article_max_len=$5
headline_max_len=$6
split=$7
seed=$8
results_name=$9

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate DL
cd /scratch/csp9835/summarization-senationalism/src

python summarization/train.py --epochs $epochs --model $model --batch_size $batch_size --learning_rate $lr --article_max_len $article_max_len --headline_max_len $headline_max_len --split $split --seed $seed --results_name $results_name
