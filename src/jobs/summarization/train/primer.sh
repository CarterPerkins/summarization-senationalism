#!/bin/bash

memory=16
duration='12:00:00'
nodes=1
cpus=4
gres='gpu:v100:1'

epochs=$1
model=$2
batch_size=$3
lr=$4
article_max_len=$5
headline_max_len=$6
split=$7
seed=$8

jobname=T_${model}_${batch_size}_${article_max_len}_${headline_max_len}

sbatch --mem=${memory}GB --time=$duration --job-name=$jobname --output=${jobname}.out --cpus-per-task=$cpus --nodes=$nodes --gres=$gres ./jobs/summarization/train/runner.sh $epochs $model $batch_size $lr $article_max_len $headline_max_len $split $seed $jobname
