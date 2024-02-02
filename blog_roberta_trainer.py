"""Train and save a distil-RoBERTa model with an expanded maximum context."""

from datasets import load_from_disk

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers.models.roberta.configuration_roberta import *
from transformers.models.roberta.modeling_roberta import *
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import EarlyStoppingCallback

import wandb

import torch
import pandas as pd
import numpy as np
import os
import time
import subprocess as sp
import copy


def sample_decode(ds, tokenizer, split='train', n=5):
    for i in range(n):
        split_ds = ds[split]
        print(tokenizer.decode(split_ds[i]['input_ids']))
        print("\n\n\n")


def describe(df):
    base = df.describe()
    base.loc["1%"] = df.quantile(0.01)
    base.loc["99%"] = df.quantile(0.99)
    return base.reindex(["count", "mean", "std", "min", "1%", "25%", "50%", "75%", "99%", "max"])


def describe_lens(ds, split='train', user=False):
    df = ds[split].to_pandas()
    if user:
        print("Not implemented yet")
    else:
        print(describe(df['input_ids'].str.len()))


class TrainParams: # all args for train() in one place
    def __init__(self, data_path, model_path, run_name, pos_embd_strat, lr, attn_dropout, weight_decay):
        self.data_path=data_path
        self.model_path=model_path
        self.run_name=run_name
        self.pos_embd_strat=pos_embd_strat
        self.lr=lr
        self.attn_dropout=attn_dropout
        self.weight_decay=weight_decay


def train(params: TrainParams):

    # load saved dataset
    print("loading dataset...")
    ds = load_from_disk(params.data_path)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # config
    config_dict = {
            'vocab_size' : 50265, # number of total tokens allowed
            'num_hidden_layers' : 6, # number of hidden RobertaLayers in a RobertaEncoder
            'num_attention_heads' : 12, # multi-headed attention heads
            'hidden_size' : 768, # dimension of hidden layers
            'intermediate_size' : 3072, # dimension of feedfoward layer in encoder
            'max_position_embeddings' : 514, # max seq. length the model could ever have
            'new_max_position_embeddings' : 4098, # max seq. length the model could ever have
            'hidden_act' : "gelu", # nonlinearity in the encoder and pooler
            'hidden_dropout_prob' : 0.1, # dropout probability for fully conn. layers
            'attention_probs_dropout_prob' : params.attn_dropout, # dropout in attn layer
            'type_vocab_size' : 1, # for 'token_type_ids' column
            'initializer_range' : 0.02, # stdev for initializing weight matrices
            'layer_norm_eps' : 1e-05, # epsilon in layer norm
            'position_embedding_type' : 'absolute', # there's special pos embds
            'bos_token_id' : 0,
            'pad_token_id' : 1,
            'eos_token_id' : 2,
            'model_type' : 'roberta',
            'is_decoder' : False, # is decoder-only
            'use_cache' : True, # return the last attn key/values
            'classifier_dropout' : 0, # dropout rate for classification head
            } 
    roberta_config = RobertaConfig(**config_dict)

    # model
    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base",
            config=roberta_config, ignore_mismatched_sizes=True)
    # expand positional embds with custom method in modeling_roberta.py
    model.expand_embds(roberta_config.new_max_position_embeddings, params.pos_embd_strat)
    

    # args
    training_args = TrainingArguments(
        output_dir = params.model_path,
        overwrite_output_dir=True,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        gradient_accumulation_steps=1,
        warmup_ratio=0.02,
        num_train_epochs=50,
        learning_rate=params.lr,
        weight_decay=params.weight_decay,
        per_device_train_batch_size=1,
        save_steps=5000,
        save_total_limit=2,
        prediction_loss_only=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
        run_name=params.run_name,
        report_to="wandb",
    )

    # data collator - performs batching and masking (i think)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # create callback for early stopping
    early_stop = EarlyStoppingCallback(early_stopping_patience=5)

    # instantiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=data_collator,
        callbacks=[early_stop],
    )
    
    # train and save
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model()
    wandb.finish()


# method to help pick a free GPU
def pick_gpu():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    for j in range(len(memory_free_values)):
        if memory_free_values[j] == 48676:
            print(f"using GPU {j}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(j)
            break


def main():
    base_data_path = "/cronus_data/ssmith/data/blogsUD/"
    base_model_path = "/cronus_data/ssmith/models/blogsUD/"
    dsep_run_name = "dsep_model"
    non_dsep_run_name = "non_dsep_model"

    dsep_train_params = TrainParams(
        data_path=base_data_path + "sample_chunked_dsep_4096",
        model_path=base_model_path + dsep_run_name,
        run_name=dsep_run_name,
        pos_embd_strat="load_repeat",
        lr=1.3e-4,
        attn_dropout=0.105,
        weight_decay=7e-3,
    )
    non_dsep_train_params = TrainParams(
        data_path=base_data_path + "sample_chunked_4096",
        model_path=base_model_path + non_dsep_run_name,
        run_name=non_dsep_run_name,
        pos_embd_strat="load_repeat",
        lr=1.3e-4,
        attn_dropout=0.105,
        weight_decay=7e-3,
    )

    # set wandb project
    os.environ["WANDB_PROJECT"] = "dsep_experiment"

    # pick GPU with memory available
    pick_gpu()

    # train
    for train_params in [dsep_train_params, non_dsep_train_params]:
        train(train_params)
        print("\n\n------\n\n------\n\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
