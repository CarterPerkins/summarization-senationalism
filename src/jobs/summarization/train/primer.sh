#!/bin/bash
mem=$1

sbatch --mem ${mem}GB \
	src/summarization/scripts/train.py ARG1 ARG2 ... ARGN