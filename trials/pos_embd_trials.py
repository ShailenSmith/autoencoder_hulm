from datasets import load_from_disk

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers.models.roberta.configuration_roberta import *
from transformers.models.roberta.modeling_roberta import *
from transformers import TrainingArguments
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
import sys


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


def run_trials(data_path, model_path, run_name, pos_embd_strat="load_512"):

    # load saved dataset
    print("loading dataset...")
    ds = load_from_disk(data_path)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # args
    args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        gradient_accumulation_steps=1,
        # warmup_ratio=0.02, # warmup ratio defined in objective() function
        num_train_epochs=20,
        per_device_train_batch_size=1,
        save_steps=5000,
        save_total_limit=2,
        prediction_loss_only=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
        report_to="wandb",
    )

    wd_to_loss = {}
    def objective(trial: optuna.Trial, args: TrainingArguments):
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
        args.run_name=f"{run_name}_{trial.number+21}"
        args.output_dir = f"{model_path}/{args.run_name}"
        args.warmup_ratio=1/args.num_train_epochs # warmup lr for 1 epoch
        args.learning_rate=3e-5 # trial.suggest_float("learning_rate", low=3e-5, high=3e-4, log=True)
        args.weight_decay=trial.suggest_categorical("weight_decay", [2e-2, 4e-2, 6e-2, 8e-2, 0.1])

        # do not run trial if weight decay is already used
        add_wd = False
        if args.weight_decay in wd_to_loss.keys():
            return wd_to_loss[args.weight_decay]
        else:
            add_wd = True
        
        # model
        model = AutoModelForMaskedLM.from_pretrained("distilroberta-base",
            config=roberta_config, ignore_mismatched_sizes=True)
        use_trained = False if pos_embd_strat == "no_load" else True
        repeat = True if pos_embd_strat == "load_repeat" else False
        model.expand_embds(roberta_config.new_max_position_embeddings, # expand positional embds with custom method in modeling_roberta.py
                use_trained=use_trained, repeat=repeat)
        print("pos embds stdev: ",
                torch.std(model.roberta.embeddings.position_embeddings.weight))

        # data collator - performs batching and masking (i think)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        # trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds['train'], #.select(np.arange(3)),
            eval_dataset=ds['dev'], #.select(np.arange(3)),
            data_collator = data_collator,
        )

        # train and evaluate
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        if add_wd:
            wd_to_loss[args.weight_decay] = eval_result['eval_loss']
        return eval_result['eval_loss']

    # sampler and study
    sampler = optuna.samplers.TPESampler() 
    study = optuna.create_study(study_name='hyper-parameter-search', direction='minimize', sampler=sampler,
                                pruner=HyperbandPruner(max_resource = args.num_train_epochs)) 

    # wandb callback and optimize 
    wandb_kwargs = {"project": os.environ["WANDB_PROJECT"]}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
    study.optimize(func=lambda trial: objective(trial, args), n_trials=20, callbacks=[wandbc])  

    print(study.best_trial)
    wandb.finish()


# method to help pick a free GPU
def pick_gpu(wait_one_gpu=False, gpu_idx=0):
    if wait_one_gpu:
        while True:
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
            if memory_free_values[gpu_idx] == 48676:
                print(f"using GPU {gpu_idx}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                return
            print(f"GPU {gpu_idx} not available, sleeping for 3 minutes")
            time.sleep(180)
    else:
        while True:
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
            for idx in range(len(memory_free_values)):
                if memory_free_values[idx] == 48676:
                    print(f"using GPU {idx}")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
                    return
            print(f"GPU {gpu_idx} not available, sleeping for 60 minutes")
            time.sleep(1800)
            print("30 minutes left")
            time.sleep(1800)

            

# ------------------- Main method ---------------------------

base_data_path = "/cronus_data/ssmith/data/blogsUD/"
base_model_path = "/cronus_data/ssmith/models/blogsUD/"
sample_chunked_path = base_data_path + "sample_chunked_4096"
sample_chunked_dsep_path = base_data_path + "sample_chunked_dsep_4096"
no_load_model_path = base_model_path + "no_load_trials"
load_repeat_model_path = base_model_path + "load_repeat_trials"
load_512_model_path = base_model_path + "load_512_trials"


data_paths = [sample_chunked_dsep_path, sample_chunked_dsep_path, sample_chunked_dsep_path]
model_paths = [no_load_model_path, load_repeat_model_path, load_512_model_path]
pos_embd_strats = ["no_load", "load_repeat", "load_512"] # "load_repeat", "load_512", or "no_load"
run_names = ["no_load_trials", "load_repeat_trials", "load_512_trials"]

idx = int(sys.argv[1])
print(idx, type(idx), run_names[idx])
pick_gpu(wait_one_gpu=False, gpu_idx=0) # pick an open GPU to use to train

# run trials
for i in [idx]:
    os.environ["WANDB_PROJECT"] = run_names[i] 
    run_trials(data_path=data_paths[i],
            model_path=model_paths[i],
            pos_embd_strat = pos_embd_strats[i],
            run_name = run_names[i],
    )
    print("\n\n------\n\n------\n\n")
    time.sleep(60)
