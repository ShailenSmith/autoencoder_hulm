from datasets import load_from_disk

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers.models.roberta.configuration_roberta import *
from transformers.models.roberta.modeling_roberta import *
from transformers import TrainingArguments, set_seed
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import EarlyStoppingCallback, TrainerCallback

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.pruners import HyperbandPruner
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


def run_trials(data_path, model_path, run_name):

    # load saved dataset
    print("loading dataset...")
    ds = load_from_disk(data_path)
    
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
        'new_max_position_embeddings' : 514, # max seq. length the model could ever have
        'hidden_act' : "gelu", # nonlinearity in the encoder and pooler
        'hidden_dropout_prob' : 0.1, # dropout probability for fully conn. layers
        'attention_probs_dropout_prob' : 0.1,
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

    # args
    args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        logging_strategy="steps",
        logging_steps=8, # low to account for high effective batch size
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_ratio=0.06,
        num_train_epochs=40,
        per_device_train_batch_size=64,
        save_steps=5000,
        save_total_limit=2,
        seed=42,
        prediction_loss_only=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
        report_to="wandb",
    )

    # model
    def model_init():
        set_seed(args.seed)
        model = AutoModelForMaskedLM.from_pretrained("distilroberta-base",
            config=roberta_config, ignore_mismatched_sizes=True)
        return model

    # data collator - performs batching and masking (i think)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1, log=False),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [2])#, 4, 8, 16, 32]),
        }

    def compute_objective(metrics) -> float:
        metrics = copy.deepcopy(metrics)
        loss = metrics.pop("eval_loss", None)
        return loss

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        data_collator = data_collator,
        )

    best_trial = trainer.hyperparameter_search(
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=5,
        compute_objective=compute_objective
    )   
    wandb.finish()


# method to help pick a free GPU
def pick_gpu():
    while True:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        for j in range(len(memory_free_values)):
            if memory_free_values[j] == 48676:
                print(f"using GPU {j}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(j)
                return
        print("No GPU available, sleeping for 10 minutes")
        time.sleep(600)
        


# ------------------- Main method ---------------------------

base_data_path = "/cronus_data/ssmith/data/blogsUD/"
base_model_path = "/cronus_data/ssmith/models/blogsUD/trials/"
sample_chunked_path = base_data_path + "sample_chunked_docss_4096"
sample_chunked_dsep_path = base_data_path + "sample_chunked_dsep_4096"
unchunked_512_path = base_data_path + f"unchunked_512_sample"
control_model_path = base_model_path + "control_trials"

os.environ["WANDB_PROJECT"] = "control_trials"


if True:
    data_paths = [unchunked_512_path]
    model_paths = [control_model_path]
    run_names = ["control_trials"]

pick_gpu() # pick an open GPU to use to train

# run trials
for i in [0]:
    print("running trials")
    run_trials(data_path=data_paths[i],
            model_path=model_paths[i],
            run_name = run_names[i],
    )
    print("\n\n------\n\n------\n\n")
    time.sleep(60)

