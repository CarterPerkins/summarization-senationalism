#!/bin/bash

memory=32
duration='12:00:00'
nodes=1
cpus=4
gres='gpu:v100:1'

params=$1
model=$2
jobname=$3

file=out/${jobname}
error=${file}.err
out=${file}.out

sbatch --mem=${memory}GB --time=$duration --job-name=$jobname --error=${error} --output=${out} --cpus-per-task=$cpus --nodes=$nodes --gres=$gres ./jobs/summarization/eval/runner.sh $params $model
