#!/bin/bash

./jobs/summarization/train/primer.sh 30 bart 8 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 30 t5 8 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 30 pegasus 8 5.6e-5 512 128 val 0
