#!/bin/bash -l
module load miniconda
conda activate lama

export TORCH_HOME="/projectnb/cs585/students/ljmiao/lama"
export PYTHONPATH="/projectnb/cs585/students/ljmiao/lama"

python3 -c "import torch; print(torch.cuda.is_available())"

cd /projectnb/cs585/students/ljmiao/lama

python3 /projectnb/cs585/students/ljmiao/lama/bin/train.py \
  -cn lama-fourier-satellite \
  location=my_dataset \
  data.batch_size=1 \
  data.num_workers=4 \
  trainer.kwargs.max_epochs=2