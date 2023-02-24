# -*- coding: utf-8 -*-
chunk_match = True

from model.prefix_encoder import PrefixEncoder

import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
)

import json

import numpy as np

import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='train_dataset.pkl', required=True,
                    help="path to train dataset")

parser.add_argument('--test_data_dir', type=str, default='shared_task-test_set-final', required=True,
                    help="path to test dataset")

parser.add_argument('--train_batch_size', type=int, default=8,
                    help="batch size during training")

parser.add_argument('--learning_rate', type=float, default=4e-06,
                    help="learning rate for the RoBERTa encoder")

parser.add_argument('--num_epoch', type=int, default=8,
                    help="number of the training epochs")

parser.add_argument('--model', type=str, default='deepset/roberta-large-squad2',
                    help="the base model name (a huggingface model)")

parser.add_argument('--seed', type=int, default=902, help="the random seed")

parser.add_argument('--pre_seq_len', type=int, default=60, help="the length of prefix tokens")

parser.add_argument('--prefix_hidden_size', type=int, default=1024, help="the hidden size of prefix tokens")

parser.add_argument('--output_dir', type=str, default='system_predictions/', help="output directory to store predictions")

args = parser.parse_args()

os.mkdir(args.output_dir)

# model class

class RobertaPrefixModelForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.subtasks = config.subtasks
        self.subtask = config.subtask

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.qa_outputs = {subtask: torch.nn.Linear(config.hidden_size, config.num_labels) for subtask in self.subtasks}
        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # self.prefix_encoder = PrefixEncoder(config)
        self.prefix_encoders = {subtask: PrefixEncoder(config) for subtask in self.subtasks}
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        # for param in self.roberta.parameters():
        # param.requires_grad = False

    def get_prompt(self, batch_size, subtask):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoders[subtask](prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size, subtask=self.subtask)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        # logits = self.qa_outputs[self.subtask](sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


checkpoint = args.model

print('checkpoint: ', checkpoint)


tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    use_fast=True
)

config = AutoConfig.from_pretrained(
    checkpoint,
    num_labels=2
)

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = examples['id']

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


import pickle



def load_from_pickle(pickle_file):
    with open(pickle_file, "rb") as pickle_in:
        return pickle.load(pickle_in)


def save_in_json(save_dict, save_file):
    with open(save_file, 'w') as fp:
        json.dump(save_dict, fp)


import random

seed = args.seed
random.seed(seed)
train_data = load_from_pickle(args.data_dir)

task_train_dic = {}
for sample in train_data:
    task = sample[-2]
    if task not in task_train_dic.keys():
        task_train_dic[task] = []
    task_train_dic[task].append(sample)

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset


def make_hgf_data(dataset):
    df = pd.DataFrame(dataset)
    dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())

    ### convert to Huggingface dataset
    return Dataset(pa.Table.from_pandas(df))


def make_example(dataset):
    new_dataset = []
    for sample in dataset:
        if sample[2]['answer_start'] == -1:
            dic = {'id': 1, 'title': 'untitled', 'context': sample[0], 'question': sample[1],
                   'answers': {'text': [], 'answer_start': []}}
        elif sample[2]['text'] == 'AUTHOR OF THE TWEET':
            dic = {'id': 1, 'title': 'untitled', 'context': sample[0], 'question': sample[1],
                   'answers': {'text': [sample[0]], 'answer_start': [sample[2]['answer_start']]}}
        else:
            dic = {'id': 1, 'title': 'untitled', 'context': sample[0], 'question': sample[1],
                   'answers': {'text': [sample[2]['text']], 'answer_start': [sample[2]['answer_start']]}}
        new_dataset.append(dic)
    return new_dataset


train_dataset = {subtask: make_example(task_train_dic[subtask]) for subtask in task_train_dic.keys()}

question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"

train_dataset = {subtask: make_hgf_data(train_dataset[subtask]) for subtask in task_train_dic.keys()}

max_seq_length = 384
pad_on_right = tokenizer.padding_side == "right"


data_collator = (
    DataCollatorWithPadding(tokenizer)
)


from torch.utils.data import DataLoader


def make_test_dataset(dataset):
    ID_dict = {}
    for sample in dataset:
        ID = sample[-1]
        if ID not in ID_dict.keys():
            ID_dict[ID] = []
        ID_dict[ID].append(sample)
    data = []
    for i in ID_dict:
        batch = ID_dict[i]
        ID = batch[0][-1]
        samples = []
        for item in batch:
            sample = {
                'answers': {'answer_start': [], 'text': []},
                'context': item[0],
                'id': ID,
                'question': item[1],
                'title': 'untitled'
            }
            samples.append(sample)
        data.append(samples)
    return data


def make_dev_example(dataset):
    new = []
    for i in dataset:
        tup = (i[0], i[1], i[-1])
        new.append(tup)
    new_dataset = []
    for sample in list(set(new)):
        dic = {'id': sample[-1], 'title': 'untitled', 'context': sample[0], 'question': sample[1],
               'answers': {'text': [], 'answer_start': []}}
        new_dataset.append(dic)
    return new_dataset


def make_predictions_on_test_dataset(model, data, tokenizer, task, threshold=0):
    model.eval()
    softmax_func = nn.Softmax(dim=0)
    sigmoid_func = nn.Sigmoid()
    model.to(device)
    task_data_dic = {}
    for sample in data:
        subtask = sample[-2]
        if subtask not in task_data_dic.keys():
            task_data_dic[subtask] = []
        task_data_dic[subtask].append(sample)
    subtasks = list(task_data_dic.keys())
    subtask_predictions = {}
    system_prediction = []
    all_ids = []
    for sample in data:
        ID = sample[-1]
        all_ids.append(ID)
    for subtask in subtasks:
        model.subtask = subtask
        ids = []
        predictions = []
        data = task_data_dic[subtask]
        data = make_dev_example(data)
        data = make_hgf_data(data)
        data = data.map(
            prepare_train_features,
            batched=True,
            remove_columns=column_names,
            desc=f"Running tokenizer on {task}: {subtask}",
        )
        ids = data['id']
        data = data.remove_columns('id')
        batch_size = 8
        data_loader = DataLoader(data, shuffle=False, collate_fn=data_collator, batch_size=batch_size)
        for batch in data_loader:
            with torch.no_grad():
                batch.to(device)
                outputs = model(**batch)
                for index in range(len(batch['input_ids'])):
                    start_logit = softmax_func(outputs['start_logits'][index]).topk(2)[1].tolist()
                    end_logit = softmax_func(outputs['end_logits'][index]).topk(2)[1].tolist()
                    input_id = batch['input_ids'][index].tolist()
                    predicted_answer = input_id[start_logit[0]:end_logit[0] + 1]
                    if len(predicted_answer) > 10:
                        predicted_answer = 'AUTHOR OF THE TWEET'
                    else:
                        predicted_answer = tokenizer.decode(predicted_answer)
                    if predicted_answer == neg_tok or predicted_answer == '':
                        if softmax_func(outputs['start_logits'][index]).topk(2)[0][0].tolist() > threshold and \
                                softmax_func(outputs['end_logits'][index]).topk(2)[0][0].tolist() > threshold:
                            predicted_answer = 'Not Specified'
                        else:
                            predicted_answer = input_id[start_logit[1]:end_logit[1] + 1]
                            if len(predicted_answer) > 10:
                                predicted_answer = 'AUTHOR OF THE TWEET'
                            else:
                                predicted_answer = tokenizer.decode(predicted_answer)
                    if len(predicted_answer) > 0 and predicted_answer[0] == ' ':
                        predicted_answer = predicted_answer[1:]
                    predictions.append(predicted_answer)
            subtask_predictions[subtask] = dict(zip(ids, predictions))
    all_ids = list(set(all_ids))
    for ID in ids:
        if task == 'positive':
            system_prediction.append({'id': ID,
                                      'predicted_annotation': {'part1.Response': ['Not Specified'],
                                                               'part2-age.Response': [subtask_predictions['age'][ID]],
                                                               'part2-close_contact.Response': [
                                                                   subtask_predictions['close_contact'][ID]],
                                                               'part2-employer.Response': [
                                                                   subtask_predictions['employer'][ID]],
                                                               'part2-gender.Response': [
                                                                   subtask_predictions['gender_male'][ID],
                                                                   subtask_predictions['gender_female'][ID]],
                                                               'part2-name.Response': [subtask_predictions['name'][ID]],
                                                               'part2-recent_travel.Response': [
                                                                   subtask_predictions['recent_travel'][ID]],
                                                               'part2-relation.Response': [
                                                                   subtask_predictions['relation'][ID]],
                                                               'part2-when.Response': [subtask_predictions['when'][ID]],
                                                               'part2-where.Response': [
                                                                   subtask_predictions['where'][ID]]}})
        if task == 'can_not_test':
            system_prediction.append({'id': ID,
                                      'predicted_annotation': {'part1.Response': ['Not Specified'],
                                                               'part2-relation.Response': [
                                                                   subtask_predictions['relation'][ID]],
                                                               'part2-symptoms.Response': [
                                                                   subtask_predictions['symptoms'][ID]],
                                                               'part2-name.Response': [subtask_predictions['name'][ID]],
                                                               'part2-when.Response': [subtask_predictions['when'][ID]],
                                                               'part2-where.Response': [
                                                                   subtask_predictions['where'][ID]]}})
        if task == 'cure':
            system_prediction.append({'id': ID,
                                      'predicted_annotation': {
                                          'part2-opinion.Response': [subtask_predictions['opinion'][ID]],
                                          'part1.Response': ['Not Specified'],
                                          'part2-what_cure.Response': [subtask_predictions['what_cure'][ID]],
                                          'part2-who_cure.Response': [subtask_predictions['who_cure'][ID]]}})
        if task == 'death':
            system_prediction.append({'id': ID,
                                      'predicted_annotation': {'part1.Response': ['Not Specified'],
                                                               'part2-age.Response': [subtask_predictions['age'][ID]],
                                                               'part2-name.Response': [subtask_predictions['name'][ID]],
                                                               'part2-relation.Response': [
                                                                   subtask_predictions['relation'][ID]],
                                                               'part2-when.Response': [subtask_predictions['when'][ID]],
                                                               'part2-where.Response': [
                                                                   subtask_predictions['where'][ID]]}})
        if task == 'negative':
            system_prediction.append({'id': ID,
                                      'predicted_annotation': {'part1.Response': ['Not Specified'],
                                                               'part2-age.Response': [subtask_predictions['age'][ID]],
                                                               'part2-close_contact.Response': [
                                                                   subtask_predictions['close_contact'][ID]],
                                                               'part2-gender.Response': [
                                                                   subtask_predictions['gender_male'][ID],
                                                                   subtask_predictions['gender_female'][ID]],
                                                               'part2-name.Response': [subtask_predictions['name'][ID]],
                                                               'part2-relation.Response': [
                                                                   subtask_predictions['relation'][ID]],
                                                               'part2-when.Response': [subtask_predictions['when'][ID]],
                                                               'part2-where.Response': [
                                                                   subtask_predictions['where'][ID]]}})
    return update_system_prediction(system_prediction, task)


def update_system_prediction(system_prediction, task):
    male_pronouns = ['he', 'him', 'his', 'father', 'brother', 'son', 'male', 'man', 'men', 'dad', 'trump']
    female_pronouns = ['her', 'she', 'mother', 'sister', 'famele', 'woman', 'women', 'lady', 'ladies', 'mom']
    for sample in system_prediction:
        first = True
        for i in sample['predicted_annotation'].values():
            if first:
                first = False
                continue
            if i != ['Not Specified']:
                sample['predicted_annotation']['part1.Response'] = ['yes']
        if 'part2-relation.Response' in sample['predicted_annotation'].keys():
            if sample['predicted_annotation']['part2-relation.Response'] != ['Not Specified']:
                sample['predicted_annotation']['part2-relation.Response'] = ['Yes']
        if 'part2-gender.Response' in sample['predicted_annotation'].keys():
            gender_prediction = sample['predicted_annotation']['part2-gender.Response']
            if gender_prediction == ['Not Specified', 'Not Specified']:
                gender_prediction = ['Not Specified']
            if len(gender_prediction) > 1 and gender_prediction[1] != 'Not Specified':
                gender_prediction = ['Female']
            if len(gender_prediction) > 1 and gender_prediction[0] != 'Not Specified':
                gender_prediction = ['Male']
            sample['predicted_annotation']['part2-gender.Response'] = gender_prediction
        if 'part2-symptoms.Response' in sample['predicted_annotation'].keys():
            if sample['predicted_annotation']['part2-symptoms.Response'] != ['Not Specified']:
                sample['predicted_annotation']['part2-symptoms.Response'] = ['Yes']
        if 'part2-opinion.Response' in sample['predicted_annotation'].keys():
            if sample['predicted_annotation']['part2-opinion.Response'] != ['Not Specified']:
                sample['predicted_annotation']['part2-opinion.Response'] = ['effective']
            else:
                sample['predicted_annotation']['part2-opinion.Response'] = ['not_effective']
    return system_prediction


def readJSONLine(path):
    output = []
    with open(path, 'r') as f:
        for line in f:
            output.append(json.loads(line))

    return output


positive_dic = {
    'age': 'What is the age of the person?',
    'close_contact': 'Who is in close contact?',
    'employer': 'Who is the employer?',
    'gender_male': 'Is the gender male?',
    'gender_female': 'Is the gender female?',
    'name': 'Who is tested positive?',
    'recent_travel': 'Where did the person recently visit?',
    'relation': 'Does the person have a relationship?',
    'when': 'When is the cases reported?',
    'where': 'Where is the cases reported?'
}
can_not_test_dic = {
    'relation': 'Does the person have a relationship?',
    'symptoms': 'Is the person experiencing any symptoms?',
    'name': 'Who can not get a test?',
    'when': 'When is the situation reported?',
    'where': 'Where is the situation reported?'
}
cure_dic = {
    'opinion': 'Does the author believe the method?',
    'what_cure': 'What is the cure?',
    'who_cure': 'Who is promoting the cure?'
}
death_dic = {
    'age': 'What is the age of the person?',
    'name': 'Who is dead?',
    'relation': 'Does the person have a relationship?',
    'when': 'When is the case reported?',
    'where': 'Where is the case reported?'
}
negative_dic = {
    'age': 'What is the age of the person?',
    'close_contact': 'Who is in close contact?',
    'gender_male': 'Is the gender male?',
    'gender_female': 'Is the gender female?',
    'name': 'Who is tested negative?',
    'relation': 'Does the person have a relationship?',
    'when': 'When is the cases reported?',
    'where': 'Where is the cases reported?'
}


def make_test_dataset(original_dataset, dic):
    data = []
    for example in original_dataset:
        tweet = example['text']
        ID = example['id']
        subtasks = list(dic.keys())
        for subtask in subtasks:
            sample = (
                tweet,
                dic[subtask],
                {'text': '', 'answer_start': -1, 'answer_end': -1},
                0,
                subtask,
                ID
            )

            data.append(sample)
    return data


# prepare test dataset
positive_ann = readJSONLine('golden/positive_sol.jsonl')
negative_ann = readJSONLine('golden/negative_sol.jsonl')
can_not_test_ann = readJSONLine('golden/can_not_test_sol.jsonl')
cure_ann = readJSONLine('golden/cure_sol.jsonl')
death_ann = readJSONLine('golden/death_sol.jsonl')

raw_positive = readJSONLine(args.test_data_dir + '/shared_task-test-positive.jsonl')
raw_can_not_test = readJSONLine(args.test_data_dir + '/shared_task-test-can_not_test.jsonl')
raw_cure = readJSONLine(args.test_data_dir + '/shared_task-test-cure.jsonl')
raw_death = readJSONLine(args.test_data_dir + '/shared_task-test-death.jsonl')
raw_negative = readJSONLine(args.test_data_dir + '/shared_task-test-negative.jsonl')

positive_dev = make_test_dataset(raw_positive, positive_dic)
can_not_test_dev = make_test_dataset(raw_can_not_test, can_not_test_dic)
cure_dev = make_test_dataset(raw_cure, cure_dic)
death_dev = make_test_dataset(raw_death, death_dic)
negative_dev = make_test_dataset(raw_negative, negative_dic)


def load_from_pickle(pickle_file):
    with open(pickle_file, "rb") as pickle_in:
        return pickle.load(pickle_in)


# make null prediction to "not specified"
def update_negatives(predictions):
    for sample in predictions:
        for task in sample['predicted_annotation'].keys():
            if sample['predicted_annotation'][task] == ['']:
                sample['predicted_annotation'][task] = ['Not Specified']


import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


# candidate chunk match
def nor_pred_chunks(curr_pred, candidate_chunks):
    nor_pred = []
    chunk_score = {}
    for candidate_chunk in candidate_chunks:
        # if curr_pred in candidate_chunk:
        # nor_pred.append(candidate_chunk)
        ptoks = get_tokens(curr_pred)
        ctoks = get_tokens(candidate_chunk)
        t = max(len(ptoks), len(ctoks))
        l = len(ptoks)
        s = 0
        for tok in ptoks:
            if tok in ctoks:
                s += 1
        if s == l:
            nor_pred.append(candidate_chunk)
            score = l / t
            chunk_score[candidate_chunk] = score

        l = len(ctoks)
        s = 0
        for tok in ctoks:
            if tok in ptoks:
                s += 1
        if s == l:
            nor_pred.append(candidate_chunk)
            score = l / t
            chunk_score[candidate_chunk] = score
    if nor_pred == []:
        nor_pred = ['Not Specified']
    else:
        nor_pred = list(set(nor_pred))
        nor_pred = [i for i in nor_pred if chunk_score[i] == max(list(chunk_score.values()))]
        if nor_pred == ['i'] or nor_pred == ['im']:
            nor_pred = ['AUTHOR OF THE TWEET']
        # nor_pred = [i for i in nor_pred if abs(len(i)-len(curr_pred))==min([abs(len(i)-len(curr_pred)) for i in nor_pred])]
    return nor_pred


def update_candidate_chunks_offsets(predictions, task):
    # golden_predictions = readJSONLine(golden_path + task + '_sol.jsonl')
    # golden_predictions_dict = {}
    # for each_line in golden_predictions:
    # golden_predictions_dict[each_line['id']] = each_line
    pre_dict = {}
    for sample in predictions:
        pre_dict[sample['id']] = sample
    task_raw_dict = {
        'positive': raw_positive,
        'can_not_test': raw_can_not_test,
        'cure': raw_cure,
        'death': raw_death,
        'negative': raw_negative
    }
    raw = task_raw_dict[task]
    raw_dict = {}
    for sample in raw:
        raw_dict[sample['id']] = sample
    for sample in predictions:
        ID = sample['id']
        candidate_chunks_offsets = raw_dict[ID]['candidate_chunks_offsets']
        candidate_chunks = [raw_dict[ID]['text'][i[0]:i[1]] for i in candidate_chunks_offsets]
        candidate_chunks = [i.lower() for i in candidate_chunks]
        for sub_task in sample['predicted_annotation'].keys():
            curr_pred = sample['predicted_annotation'][sub_task][0]
            if curr_pred != 'Not Specified' and curr_pred != 'effective' and curr_pred != 'yes' and curr_pred != 'not_effective' and curr_pred != 'AUTHOR OF THE TWEET' and curr_pred != 'Male' and curr_pred != 'Female' and curr_pred != 'Yes':
                curr_pred = curr_pred.lower()
                if curr_pred not in candidate_chunks:
                    if 'and' in curr_pred:
                        curr_preds = curr_pred.split(' and ')
                        nor_pred = []
                        for pred in curr_preds:
                            sub_pred = nor_pred_chunks(pred, candidate_chunks)
                            if sub_pred != ['Not Specified']:
                                nor_pred = nor_pred + sub_pred
                    elif ',' in curr_pred:
                        curr_preds = curr_pred.split(', ')
                        nor_pred = []
                        for pred in curr_preds:
                            sub_pred = nor_pred_chunks(pred, candidate_chunks)
                            if sub_pred != ['Not Specified']:
                                nor_pred = nor_pred + sub_pred
                    else:
                        nor_pred = nor_pred_chunks(curr_pred, candidate_chunks)
                    nor_pred = list(set(nor_pred))
                    sample['predicted_annotation'][sub_task] = nor_pred


def runEvaluation(system_predictions, golden_predictions):
    ## read in files
    golden_predictions_dict = {}
    for each_line in golden_predictions:
        golden_predictions_dict[each_line['id']] = each_line

    ## question tags
    question_tag = [i for i in golden_predictions[0]['golden_annotation'] if 'part2' in i]

    ## evaluation
    result = {}
    for each_task in question_tag:

        # evaluate curr task
        curr_task = {}
        TP, FP, FN = 0.0, 0.0, 0.0
        for each_line in system_predictions:
            curr_sys_pred = [i.lower() for i in each_line['predicted_annotation'][each_task] if \
                             i != 'Not Specified' and i != 'not specified' and i != 'not_effective']
            #             print(golden_predictions_dict[each_line['id']]['golden_annotation'][each_task])
            curr_golden_ann = [i.lower() for i in
                               golden_predictions_dict[each_line['id']]['golden_annotation'][each_task] \
                               if i != 'Not Specified' and i != 'not specified' and i != 'not_effective']
            #             print(curr_sys_pred, curr_golden_ann)
            if len(curr_golden_ann) > 0:
                for predicted_chunk in curr_sys_pred:
                    if predicted_chunk in curr_golden_ann:
                        TP += 1  # True positives are predicted spans that appear in the gold labels.
                    else:
                        FP += 1  # False positives are predicted spans that don't appear in the gold labels.
                for gold_chunk in curr_golden_ann:
                    if gold_chunk not in curr_sys_pred:
                        FN += 1  # False negatives are gold spans that weren't in the set of spans predicted by the model.
            else:
                if len(curr_sys_pred) > 0:
                    for predicted_chunk in curr_sys_pred:
                        FP += 1  # False positives are predicted spans that don't appear in the gold labels.

        # print
        if TP + FP == 0:
            P = 0.0
        else:
            P = TP / (TP + FP)

        if TP + FN == 0:
            R = 0.0
        else:
            R = TP / (TP + FN)

        if P + R == 0:
            F1 = 0.0
        else:
            F1 = 2.0 * P * R / (P + R)

        curr_task["F1"] = F1
        curr_task["P"] = P
        curr_task["R"] = R
        curr_task["TP"] = TP
        curr_task["FP"] = FP
        curr_task["FN"] = FN
        N = TP + FN
        curr_task["N"] = N

        # print(curr_task)
        result[each_task.replace('.Response', '')] = curr_task

        # print
    #         print(each_task.replace('.Response', ''))
    #         print('P:', curr_task['P'], 'R:', curr_task['R'], 'F1:', curr_task['F1'])
    #         print('=======')

    ### calculate micro-F1
    all_TP = np.sum([i[1]['TP'] for i in result.items()])
    all_FP = np.sum([i[1]['FP'] for i in result.items()])
    all_FN = np.sum([i[1]['FN'] for i in result.items()])

    all_P = all_TP / (all_TP + all_FP)
    all_R = all_TP / (all_TP + all_FN)
    all_F1 = 2.0 * all_P * all_R / (all_P + all_R)

    ## append
    result['micro'] = {}
    result['micro']['TP'] = all_TP
    result['micro']['FP'] = all_FP
    result['micro']['FN'] = all_FN
    result['micro']['P'] = all_P
    result['micro']['R'] = all_R
    result['micro']['F1'] = all_F1
    result['micro']['N'] = all_TP + all_FN

    #     print('micro F1', all_F1)

    return result


from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup

accelerator = Accelerator()

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
epochs = args.num_epoch
batch_size = args.train_batch_size

# prepare training dataset
train_dataloader_dic = {}
for subtask, subtask_data in train_dataset.items():
    column_names = subtask_data.column_names
    subtask_data = subtask_data.map(
        prepare_train_features,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )
    train_dataloader_dic[subtask] = DataLoader(subtask_data, shuffle=True, collate_fn=data_collator,
                                               batch_size=batch_size)

# define model and hyper-parameters
config.subtask = None
config.subtasks = list(train_dataloader_dic.keys())
config.hidden_dropout_prob = 0.2
config.pre_seq_len = args.pre_seq_len
config.prefix_projection = True
config.prefix_hidden_size = args.prefix_hidden_size

model_class = RobertaPrefixModelForQuestionAnswering

model = model_class.from_pretrained(
    checkpoint,
    config=config
).to(device)

if model_class == RobertaPrefixModelForQuestionAnswering:
    for subtask, prefix_encoder in model.prefix_encoders.items():
        prefix_encoder.to(device)

neg_tok = '<s>'

total_steps = sum([len(train_dataloader_dic[subtask]) for subtask in train_dataset.keys()]) * epochs
gradient_accumulation_steps = 1
lr = args.learning_rate
optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

import time
import datetime
from tqdm.auto import tqdm


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

total_start_time = time.time()
epoch_train_loss = list()

train_loss_trajectory = list()
start_time = time.time()
print('seed =', seed)
print('epochs =', epochs)
print('batch_size =', batch_size)
print('gradient_accumulation_steps =', gradient_accumulation_steps)
print('learning rate =', lr)
if model_class == RobertaPrefixModelForQuestionAnswering:
    print('pre_seq_len =', model.pre_seq_len)
    print('prefix_hidden_size =', config.prefix_hidden_size)

    
# let's start training
for epoch in range(epochs):
    for subtask, train_dataloader in train_dataloader_dic.items():
        # model config
        model.subtask = subtask
        # stop gradient for other subtasks
        if model_class == RobertaPrefixModelForQuestionAnswering:
            for st in model.subtasks:
                if st != subtask:
                    for param in model.prefix_encoders[st].parameters():
                        param.requires_grad = False
                        param.grad = None

                else:
                    for param in model.prefix_encoders[st].parameters():
                        param.requires_grad = True

        pbar = tqdm(train_dataloader)
        elapsed = format_time(time.time() - start_time)
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            total_train_loss += loss.item()
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                avg_train_loss = total_train_loss / (step + 1)
                train_loss_trajectory.append(avg_train_loss)
                pbar.set_description(
                    f"Epoch:{epoch + 1}|Batch:{step}/{len(train_dataloader)}|Time:{elapsed}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

    if epoch == epochs - 1:
        positive_prediction = make_predictions_on_test_dataset(model, positive_dev, tokenizer, 'positive')
        can_not_test_prediction = make_predictions_on_test_dataset(model, can_not_test_dev, tokenizer, 'can_not_test')
        cure_prediction = make_predictions_on_test_dataset(model, cure_dev, tokenizer, 'cure')
        death_prediction = make_predictions_on_test_dataset(model, death_dev, tokenizer, 'death')
        negative_prediction = make_predictions_on_test_dataset(model, negative_dev, tokenizer, 'negative')

        if chunk_match:
            update_candidate_chunks_offsets(positive_prediction, 'positive')
            update_candidate_chunks_offsets(negative_prediction, 'negative')
            update_candidate_chunks_offsets(can_not_test_prediction, 'can_not_test')
            update_candidate_chunks_offsets(cure_prediction, 'cure')
            update_candidate_chunks_offsets(death_prediction, 'death')

            update_negatives(positive_prediction)
            update_negatives(negative_prediction)
            update_negatives(can_not_test_prediction)
            update_negatives(cure_prediction)
            update_negatives(death_prediction)

        save_in_json(positive_prediction, args.output_dir + 'system_predictions-positive.jsonl')
        save_in_json(can_not_test_prediction, args.output_dir + 'system_predictions-can_not_test.jsonl')
        save_in_json(cure_prediction, args.output_dir + 'system_predictions-cure.jsonl')
        save_in_json(death_prediction, args.output_dir + 'system_predictions-death.jsonl')
        save_in_json(negative_prediction, args.output_dir + 'system_predictions-negative.jsonl')

        curr_team = {}
        category_flag = ['positive', 'can_not_test', 'cure', 'death', 'negative']
        all_category_results = {}
        curr_result = runEvaluation(positive_prediction, positive_ann)
        all_category_results[category_flag[0]] = curr_result
        curr_result = runEvaluation(can_not_test_prediction, can_not_test_ann)
        all_category_results[category_flag[1]] = curr_result
        curr_result = runEvaluation(cure_prediction, cure_ann)
        all_category_results[category_flag[2]] = curr_result
        curr_result = runEvaluation(death_prediction, death_ann)
        all_category_results[category_flag[3]] = curr_result
        curr_result = runEvaluation(negative_prediction, negative_ann)
        all_category_results[category_flag[4]] = curr_result

        all_cate_TP = np.sum([i[1]['micro']['TP'] for i in all_category_results.items()])
        all_cate_FP = np.sum([i[1]['micro']['FP'] for i in all_category_results.items()])
        all_cate_FN = np.sum([i[1]['micro']['FN'] for i in all_category_results.items()])

        ### micro-F1
        all_cate_P = all_cate_TP / (all_cate_TP + all_cate_FP)
        all_cate_R = all_cate_TP / (all_cate_TP + all_cate_FN)
        all_cate_F1 = 2.0 * all_cate_P * all_cate_R / (all_cate_P + all_cate_R)

        curr_team['category_perf'] = all_category_results
        merged_performance = {}
        merged_performance['TP'] = all_cate_TP
        merged_performance['FP'] = all_cate_FP
        merged_performance['FN'] = all_cate_FN
        merged_performance['P'] = all_cate_P
        merged_performance['R'] = all_cate_R
        merged_performance['F1'] = all_cate_F1
        curr_team['overall_perf'] = merged_performance
        F1 = all_cate_F1
    
        print('Fscore: ', F1)
        print('Precision: ', all_cate_P)
        print('Recall: ', all_cate_R)
