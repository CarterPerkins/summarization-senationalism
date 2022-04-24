#!/bin/bash

./jobs/summarization/train/primer.sh 1 bart 16 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 1 t5 16 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 16 5.6e-5 256 128 val 0

./jobs/summarization/train/primer.sh 1 bart 16 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 1 t5 16 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 16 5.6e-5 512 128 val 0

./jobs/summarization/train/primer.sh 1 bart 16 5.6e-5 1024 128 val 0
./jobs/summarization/train/primer.sh 1 t5 16 5.6e-5 1024 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 16 5.6e-5 1024 128 val 0

./jobs/summarization/train/primer.sh 1 bart 64 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 1 t5 64 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 64 5.6e-5 256 128 val 0

./jobs/summarization/train/primer.sh 1 bart 64 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 1 t5 64 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 64 5.6e-5 512 128 val 0

./jobs/summarization/train/primer.sh 1 bart 64 5.6e-5 1024 128 val 0
./jobs/summarization/train/primer.sh 1 t5 64 5.6e-5 1024 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 64 5.6e-5 1024 128 val 0

./jobs/summarization/train/primer.sh 1 bart 128 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 1 t5 128 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 128 5.6e-5 256 128 val 0

./jobs/summarization/train/primer.sh 1 bart 128 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 1 t5 128 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 128 5.6e-5 512 128 val 0

./jobs/summarization/train/primer.sh 1 bart 128 5.6e-5 1024 128 val 0
./jobs/summarization/train/primer.sh 1 t5 128 5.6e-5 1024 128 val 0
./jobs/summarization/train/primer.sh 1 pegasus 128 5.6e-5 1024 128 val 0
