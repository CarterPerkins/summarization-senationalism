#!/bin/bash

./jobs/summarization/train/primer.sh 30 bart 8 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 30 t5 8 5.6e-5 256 128 val 0
./jobs/summarization/train/primer.sh 30 pegasus 8 5.6e-5 256 128 val 0

./jobs/summarization/train/primer.sh 30 bart 8 5.6e-5 512 128 val 0
./jobs/summarization/train/primer.sh 30 t5 8 5.6e-5 512 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 8 5.6e-5 512 128 val 0

# Did not run in time
#./jobs/summarization/train/primer.sh 30 bart 8 5.6e-5 1024 128 val 0
#./jobs/summarization/train/primer.sh 30 t5 8 5.6e-5 1024 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 8 5.6e-5 1024 128 val 0


# Too large of batch size
#./jobs/summarization/train/primer.sh 30 bart 16 5.6e-5 256 128 val 0
#./jobs/summarization/train/primer.sh 30 t5 16 5.6e-5 256 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 16 5.6e-5 256 128 val 0

#./jobs/summarization/train/primer.sh 30 bart 16 5.6e-5 512 128 val 0
#./jobs/summarization/train/primer.sh 30 t5 16 5.6e-5 512 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 16 5.6e-5 512 128 val 0

#./jobs/summarization/train/primer.sh 30 bart 16 5.6e-5 1024 128 val 0
#./jobs/summarization/train/primer.sh 30 t5 16 5.6e-5 1024 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 16 5.6e-5 1024 128 val 0

# Too large of batch size
#./jobs/summarization/train/primer.sh 30 bart 32 5.6e-5 256 128 val 0
#./jobs/summarization/train/primer.sh 30 t5 32 5.6e-5 256 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 32 5.6e-5 256 128 val 0

#./jobs/summarization/train/primer.sh 30 bart 32 5.6e-5 512 128 val 0
#./jobs/summarization/train/primer.sh 30 t5 32 5.6e-5 512 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 32 5.6e-5 512 128 val 0

#./jobs/summarization/train/primer.sh 30 bart 32 5.6e-5 1024 128 val 0
#./jobs/summarization/train/primer.sh 30 t5 32 5.6e-5 1024 128 val 0
#./jobs/summarization/train/primer.sh 30 pegasus 32 5.6e-5 1024 128 val 0
