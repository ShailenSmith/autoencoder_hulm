from datasets import load_from_disk, Dataset, DatasetDict

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers.models.roberta.configuration_roberta import *
from transformers.models.roberta.modeling_roberta import *
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import EarlyStoppingCallback, TrainerCallback

import wandb
import torch
import pandas as pd
import os
import time
import subprocess as sp
import copy
import logging
import json
import numpy as np


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


def df_to_ds(train_df, val_df, test_df=None):    
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    ds = DatasetDict()
    ds['train'] = train_ds
    ds['validation'] = val_ds
    if test_df is not None:
        ds['test'] = Dataset.from_pandas(test_df, preserve_index=False)
    return ds


def separate_blogs_ud(ds): # deconstruct ds into (blogs_ds, ud_ds)
    blogs_dfs = {}
    ud_dfs = {}
    for split in ds.keys():
        split_df = ds[split].to_pandas()
        split_df['user_id'] = split_df['user_id'].apply(int)
        blogs_dfs[split] = split_df[split_df['user_id'] >= 5000] # min blogs user_id is 5114
        ud_dfs[split] = split_df[split_df['user_id'] < 5000] # max ud user_id is 2333

    blogs_ds = df_to_ds(*[blogs_dfs[split] for split in blogs_dfs.keys()])
    ud_ds = df_to_ds(*[ud_dfs[split] for split in ud_dfs.keys()])
    return blogs_ds, ud_ds


def evaluate(data_path, model_path):

    # load saved dataset
    print("loading dataset...")
    ds = load_from_disk(data_path)

    # split into blogs and UD datasets
    blogs_ds, ud_ds = separate_blogs_ud(ds)
    dses = {}
    dses["both"] = ds
    dses["blogs"] = blogs_ds
    dses["ud"] = ud_ds
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # args
    training_args = TrainingArguments(
        output_dir=model_path + "_eval",
        overwrite_output_dir=False,
        evaluation_strategy="epoch",
        report_to="wandb",
    )
    
    # config
    config_dict = {
        'vocab_size' : 50265, # number of total tokens allowed
        'num_hidden_layers' : 6, # number of hidden RobertaLayers in a RobertaEncoder
        'num_attention_heads' : 12, # multi-headed attention heads
        'hidden_size' : 768, # dimension of hidden layers
        'intermediate_size' : 3072, # dimension of feedfoward layer in encoder
        'max_position_embeddings' : 4098, # max seq. length the model could ever have
        'new_max_position_embeddings' : 4098, # max seq. length the model could ever have
        'hidden_act' : "gelu", # nonlinearity in the encoder and pooler
        'hidden_dropout_prob' : 0.1, # dropout probability for fully conn. layers
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
    }   
    roberta_config = RobertaConfig(**config_dict)
    
    # load trained model
    model = RobertaForMaskedLM.from_pretrained(model_path, config=roberta_config)

    # data collator - performs batching and masking (i think)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    eval_results = {}

    for domain, ds in dses.items():
        for split in ["validation", "test"]:
            run_name = f"{domain}_{split}"
            print(f" --- {run_name} ---")
            training_args.run_name = run_name
            trainer = Trainer(
                        model=model,
                        args=training_args,
                        eval_dataset=ds[split], #.select(np.arange(3)),
                        data_collator = data_collator,
                    )
            # evaluate
            eval_results[run_name] = trainer.evaluate()

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


# ------------------- Main method ---------------------------

base_data_path = "/cronus_data/ssmith/data/blogsUD/"
base_model_path = "/cronus_data/ssmith/models/blogsUD/"
sample_chunked_path = base_data_path + "sample_chunked_4096"
sample_chunked_dsep_path = base_data_path + "sample_chunked_dsep_4096"
dsep_model_path = base_model_path + "dsep_trials/dsep_trials_0/checkpoint-18220"
normal_model_path = base_model_path + "non_dsep_trials"
wandb_test_path = base_model_path + "steps_test"

os.environ["WANDB_PROJECT"] = "hulm_evaluations"

if True:
    data_paths = [sample_chunked_dsep_path]
    model_paths = [dsep_model_path]

pick_gpu() # pick an open GPU to use to train

# evaluate
for i in range(len(data_paths)):
    evaluate(data_path=data_paths[i],
            model_path=model_paths[i],
    )
    print("\n\n------\n\n------\n\n")
    time.sleep(60)
