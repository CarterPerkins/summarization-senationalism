#!/bin/bash

memory=32
duration='24:00:00'
nodes=1
cpus=4
gres='gpu:rtx8000:1'

epochs=$1
model=$2
batch_size=$3
lr=$4
article_max_len=$5
headline_max_len=$6
split=$7
seed=$8

jobname=T_${model}_${batch_size}_${article_max_len}_${headline_max_len}

file='out/${jobname}'
error='${file}.err'
out='${file}.out'

sbatch --mem=${memory}GB --time=$duration --job-name=$jobname --error=${error} --output=${out} --cpus-per-task=$cpus --nodes=$nodes --gres=$gres ./jobs/summarization/train/runner.sh $epochs $model $batch_size $lr $article_max_len $headline_max_len $split $seed $file
