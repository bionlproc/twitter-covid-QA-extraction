# COVID-19 event extraction from Twitter via extractive question answering with continuous prompts
The corresponding code for our paper: COVID-19 event extraction from Twitter via extractive question answering with continuous prompts.
## Installation
This repository requires Python 3.8 or later.
### Installing the library and dependencies
```
pip install -r requirements.txt
```
## Training & evaluation
Simply use `run.py` file and pass the hyper-parameters,
```
python run.py\
          --data_dir train_dataset.pkl\
          --train_batch_size 8\
          --learning_rate 4e-06\
          --num_epoch 8\
          --seed 902\
```
Here we train our model on the preprocessed data file `train_dataset.pkl` and evaluate on `shared_task-test_set-final`, which is a 2020 shared task on COVID-19 event extraction.
