export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)


python3 bin/train.py \
  -cn lama-fourier-satellite \
  location=my_dataset \
  data.batch_size=8 \
  data.num_workers=4 \
  trainer.kwargs.max_epochs=2 > output.txt