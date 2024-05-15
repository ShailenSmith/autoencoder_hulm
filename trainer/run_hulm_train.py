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


class MyTrainParams: # all args for run() in one place
    def __init__(self, data_path, model_path, run_name, pos_embd_strat, lr,
                epochs, weight_decay, logging_steps=500, warmup=None,
                loading_model=False):
        self.data_path=data_path
        self.model_path=model_path
        self.run_name=run_name
        self.extend_pos_embds=extend_pos_embds
        self.pos_embd_strat=pos_embd_strat
        self.lr=lr
        self.warmup=warmup
        self.epochs=epochs
        self.sched_type=sched_type
        self.weight_decay=weight_decay
        self.loading_model=loading_model


def run(params: MyTrainParams):

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
            'attention_probs_dropout_prob' : 0.1, # dropout in attn layer
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
    if extend_pos_embds:
        use_trained = False if params.pos_embd_strat == "no_load" else True
        repeat = True if params.pos_embd_strat == "load_repeat" else False
        model.expand_embds(roberta_config.new_max_position_embeddings, # expand positional embds with custom method in modeling_roberta.py
                use_trained=use_trained, repeat=repeat)

    
    # args
    training_args = TrainingArguments(
        output_dir = params.model_path,
        overwrite_output_dir=False,
        logging_strategy="steps",
        logging_steps=params.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=params.epochs,
        learning_rate=params.lr,
        weight_decay=params.weight_decay,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_steps=5000,
        save_total_limit=3,
        prediction_loss_only=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
        run_name=params.run_name,
        report_to="wandb",
    )

    if params.loading_model:
        training_args.output_dir += "/ctd"


    if params.warmup is not None:
        training_args.warmup_ratio = params.warmup

    # data collator - performs batching and masking (i think)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # create callback for early stopping
    early_stop = EarlyStoppingCallback(early_stopping_patience=5)

    # instantiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        data_collator=data_collator,
        callbacks=[early_stop],
    )
    
    # train and save
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model()
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
            print(f"GPUs not available, sleeping for 60 minutes")
            time.sleep(1800)
            print("30 minutes left")
            time.sleep(1800)


def main():
    base_data_path = "/cronus_data/ssmith/data/blogsUD/"
    base_model_path = "/cronus_data/ssmith/models/blogsUD/"
    no_load_run_name = "no_load_model"
    load_512_run_name = "load_512_model"
    load_repeat_run_name = "basic_hulm_model"

    # no_load_train_params = MyTrainParams(
    #     data_path=base_data_path + "sample_chunked_dsep_4096",
    #     model_path=base_model_path + no_load_run_name,
    #     run_name=no_load_run_name,
    #     pos_embd_strat="no_load",
    #     lr=3e-5,
    #     weight_decay=8e-2,
    # )
    # load_512_train_params = MyTrainParams(
    #     data_path=base_data_path + "sample_chunked_dsep_4096",
    #     model_path=base_model_path + load_512_run_name,
    #     run_name=load_512_run_name,
    #     pos_embd_strat="load_512",
    #     lr=3e-5,
    #     weight_decay=8e-2,
    # )

    # train 50 epoch model with 6% warmup and linear decay
    load_repeat_train_params = MyTrainParams(
        data_path=base_data_path + "chunked_dsep_4096",
        model_path=base_model_path + "basic_hulm_model_constant_lr",
        run_name="basic_hulm_model_constant_lr",
        pos_embd_strat="load_repeat",
        lr=1.2e-4,
        # warmup=0,
        epochs=50,
        sched_type="constant",
        weight_decay=8e-2,
        resume_from_checkpoint=False,
        loading_model=False,
    )


    # make sure debugging in Trainer is turned off
    os.environ["TESTING"] = "False"

    # set wandb project
    os.environ["WANDB_PROJECT"] = "basic_hulm_train"

    # pick GPU with memory available
    pick_gpu(wait_one_gpu=False, gpu_idx=0)

    # train
    for train_params in [load_repeat_train_params]:
        train(train_params)
        print("\n\n------\n\n------\n\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
