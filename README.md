# COVID-19 event extraction from Twitter via extractive question answering with continuous prompts
The corresponding code for our paper: COVID-19 event extraction from Twitter via extractive question answering with continuous prompts.
## Installation
This repository requires Python 3.8 or later.
### Installing the library and dependencies
```
pip install -r requirements.txt
```
## Training & evaluation
Here we train our model on the preprocessed data file `train_dataset.pkl` and evaluate on `shared_task-test_set-final`, which is a 2020 shared task on COVID-19 event extraction. Simply use `run.py` file,
```
python run.py\
          --data_dir train_dataset.pkl\
          --train_batch_size 8\
          --learning_rate 4e-06\
          --num_epoch 8\
          --seed 902\
```

```
usage: run.py [-h] --data_dir DATA_DIR --test_data_dir TEST_DATA_DIR
              [--train_batch_size TRAIN_BATCH_SIZE]
              [--learning_rate LEARNING_RATE] [--num_epoch NUM_EPOCH]
              [--model MODEL] [--seed SEED] [--pre_seq_len PRE_SEQ_LEN]
              [--prefix_hidden_size PREFIX_HIDDEN_SIZE]
              [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   path to train dataset
  --test_data_dir TEST_DATA_DIR
                        path to test dataset
  --train_batch_size TRAIN_BATCH_SIZE
                        batch size during training
  --learning_rate LEARNING_RATE
                        learning rate for the RoBERTa encoder
  --num_epoch NUM_EPOCH
                        number of the training epochs
  --model MODEL         the base model name (a huggingface model)
  --seed SEED           the random seed
  --pre_seq_len PRE_SEQ_LEN
                        the length of prefix tokens
  --prefix_hidden_size PREFIX_HIDDEN_SIZE
                        the hidden size of prefix tokens
  --output_dir OUTPUT_DIR
                        output directory to store predictions
```

## Dataset
To obtain the original dataset, please see [Extracting COVID-19 Events from Twitter](https://github.com/viczong/extract_COVID19_events_from_Twitter).

Preprocess the dataset to our desired format
```
python preprocess_dataset.py
```
