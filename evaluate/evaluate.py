import os
import sys
os.environ["TESTING"] = str(sys.argv[1]) if len(sys.argv) >= 2 else "False"
os.environ["DEBUG"] = os.environ["TESTING"]
GTPL = str(sys.argv[2]) if len(sys.argv) >= 3 else "False"
os.environ["GTPL"] = GTPL
custom_mask = str(sys.argv[3]).lower() in ('true', '1', 't') if len(sys.argv) >= 4 else "False"
os.environ["CUSTOM_MASK"] = str(custom_mask)


# args: (DEBUG, GTPL (get target probs and labels), CUSTOM_MASK)

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
import pickle
import time
import subprocess as sp
import copy
import logging
import json
import numpy as np
import pdb


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
    ds['dev'] = val_ds
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


def evaluate(data_path, model_path, run_name, extend_pos_embds=False, predict=False, preds_split="test", ootb=False):

    # load saved dataset
    print("loading dataset...")
    ds = load_from_disk(data_path)

    # split into blogs and UD datasets
    blogs_ds, ud_ds = separate_blogs_ud(ds)
    dses = {}
    dses["blogs_ud"] = ds
    dses["blogs"] = blogs_ds
    dses["ud"] = ud_ds
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # args
    training_args = TrainingArguments(
        output_dir=model_path + "_eval",
        overwrite_output_dir=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        # eval_accumulation_steps=15 if predict else None, # to not cause RAM issues during prediction
        evaluation_strategy="epoch",
        report_to="wandb",
        seed=42,
        include_inputs_for_metrics=True,
        run_name=run_name,
    )
    
    # config
    config_dict = {
        'vocab_size' : 50265, # number of total tokens allowed
        'num_hidden_layers' : 6, # number of hidden RobertaLayers in a RobertaEncoder
        'num_attention_heads' : 12, # multi-headed attention heads
        'hidden_size' : 768, # dimension of hidden layers
        'intermediate_size' : 3072, # dimension of feedfoward layer in encoder
        'max_position_embeddings' : 4098 if extend_pos_embds else 514, # max seq. length the model could ever have
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
    
    # load model
    if ootb: # out of the box
        model = AutoModelForMaskedLM.from_pretrained("distilroberta-base",
        config=roberta_config, ignore_mismatched_sizes=True)
    else:
        model = RobertaForMaskedLM.from_pretrained(model_path, config=roberta_config)


    # data collator - performs batching and masking (i think)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    eval_results = {}

    if predict and GTPL:
        trainer = Trainer(
                            model=model,
                            args=training_args,
                            eval_dataset=ds[preds_split],
                            data_collator=data_collator,
                        )


        if custom_mask: # the CUSTOM_MASK env variable with the same boolean value will make the necessary changes in the trainer.predict() source code
            target_probs, new_labels, new_masked_input_ids = trainer.predict(trainer.eval_dataset, return_only_inputs_labels=True)

        else:
            # get target probs from prediction with modified trainer when GTPL=True
            target_probs, labels, masked_input_ids = trainer.predict(trainer.eval_dataset, return_only_inputs_labels=True)

        target_probs = [[elt.item() for elt in lst] for lst in target_probs] # (num_rows, context_length, 1) --> (num_rows, context_length)

        if not custom_mask:
            # add masked inputs, labels, preds to ds[preds_split]
            ds[preds_split] = trainer.eval_dataset.add_column('masked_input_ids', list(masked_input_ids))
            ds[preds_split] = ds[preds_split].add_column('kept_labels', list(labels))
            ds[preds_split] = ds[preds_split].add_column('probs', list(target_probs))
            print('none over'); exit()
            ds.save_to_disk(data_path + f"_{preds_split.upper()}_WPROBS")
        else: # custom_mask
            ds[preds_split] = trainer.eval_dataset.add_column('new_masked_input_ids', list(new_masked_input_ids))
            ds[preds_split] = ds[preds_split].add_column('new_labels', list(new_labels))
            ds[preds_split] = ds[preds_split].add_column('probs', list(target_probs))
            # pdb.set_trace()
            # ds.save_to_disk(data_path + f"_{preds_split.upper()}_WPROBS")
            return ds[preds_split]
        
    elif predict:
        print("GTPL is False, not currently implemented")

    else: # evaluate
        trainer = Trainer(
                    model=model,
                    args=training_args,
                    eval_dataset=ds[preds_split].select(np.arange(75)),
                    data_collator=data_collator,
                )
        # evaluate
        eval_results = trainer.evaluate()
        pdb.set_trace()


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
base_model_path = "/cronus_data/ssmith/models/blogsUD/"
sample_chunked_dsep_path = base_data_path + "sample_chunked_dsep_4096"
sample_chunked_docss_path = base_data_path + "sample_chunked_docss_4096"
chunked_docss_path = base_data_path + f"chunked_docss_4096"
chunked_dsep_path = base_data_path + f"chunked_dsep_4096"
chunked_dsep_wlabels_path = chunked_dsep_path + "_WLABELS"
unchunked_512_path = base_data_path + f"unchunked_512"
unchunked_512_wlabels_path = unchunked_512_path + "_WLABELS"
dsep_model_path = base_model_path + "dsep_model/checkpoint-40995"
docss_model_path = base_model_path + "docss_model/checkpoint-26419"
wandb_test_path = base_model_path + "steps_test"


# our final data and models for end of my senior thesis
HULM_DATA_PATH = chunked_dsep_path
CONTROL_DATA_WLABELS_PATH = base_data_path + "_extra_probs_dses/unchunked_512_DEVTEST_WLABELS"

CONTROL_MODEL_PATH = base_model_path + "control_model/checkpoint-8800"
MAY_CONTROL_MODEL_PATH = base_model_path + "????"
HULM_MODEL_PATH = base_model_path + "basic_hulm_model_50epochs/checkpoint-906550"


os.environ["WANDB_PROJECT"] = "EVAL"

pick_gpu() # pick an open GPU to use to train


# out of the box compute target probs

if True: # control compute target probs for dev
    ootb_dev_probs_ds = evaluate(CONTROL_DATA_WLABELS_PATH,
    CONTROL_MODEL_PATH,
    "dev-control-insert-masks-OOTB",
    extend_pos_embds=False,
    predict=True,
    preds_split="dev",
    ootb=True,
    )

    ootb_test_probs_ds = evaluate(CONTROL_DATA_WLABELS_PATH,
    CONTROL_MODEL_PATH,
    "test-control-insert-masks-OOTB",
    extend_pos_embds=False,
    predict=True,
    preds_split="test",
    ootb=True,
    )


    control_probs_ds = DatasetDict()
    control_probs_ds['dev'] = ootb_dev_probs_ds
    control_probs_ds['test'] = ootb_test_probs_ds
    assert control_probs_ds['test']['user_id'] is not None
    control_probs_ds.save_to_disk(base_data_path + "OOTB_probs_ds")

if False: # non hulm (control) dev run -- MAY
    dev_probs_ds = evaluate(CONTROL_DATA_WLABELS_PATH,
    MAY_CONTROL_MODEL_PATH,
    "dev-control-insert-masks-MAY",
    extend_pos_embds=False,
    predict=True,
    preds_split="dev")

    test_probs_ds = evaluate(CONTROL_DATA_WLABELS_PATH,
    MAY_CONTROL_MODEL_PATH,
    "test-control-insert-masks-MAY",
    extend_pos_embds=False,
    predict=True,
    preds_split="test")


    control_probs_ds = DatasetDict()
    control_probs_ds['dev'] = dev_probs_ds
    control_probs_ds['test'] = test_probs_ds
    assert control_probs_ds['test']['user_id'] is not None
    control_probs_ds.save_to_disk(base_data_path + "MAY_control_probs_ds")


if False: # evaluate control
    evaluate(CONTROL_DATA_WLABELS_PATH,
    CONTROL_MODEL_PATH,
    "calc-ppl-test-control",
    extend_pos_embds=False,
    predict=False,
    preds_split="test")


if False: # evaluate hulm
    evaluate(HULM_DATA_PATH,
    HULM_MODEL_PATH,
    "calc-ppl-test-hulm",
    extend_pos_embds=True,
    predict=False,
    preds_split="test")




# hulm compute target probs for test
if False:
    evaluate(chunked_dsep_path,
    HULM_MODEL_PATH,
    "get-target-probs-test-hulm",
    extend_pos_embds=True,
    predict=True,
    preds_split="test",
    )


# hulm compute target probs for dev
if False:
    evaluate(chunked_dsep_path,
    HULM_MODEL_PATH,
    "get-target-probs-dev-hulm",
    extend_pos_embds=True,
    predict=True,
    preds_split="dev",
    )


# hulm get dev masked_inputs and labels from docss
if False:
    evaluate(chunked_docss_path,
    HULM_MODEL_PATH,
    "get-masks-labels-dev-hulm",
    extend_pos_embds=True,
    predict=True,
    preds_split="dev",
    )
    exit()


if False: # hulm test run
    evaluate(HULM_DATA_WLABELS_PATH,
    HULM_MODEL_PATH,
    "test-hulm-insert-masks",
    extend_pos_embds=True,
    predict=True)

if False: # hulm dev run
    evaluate(HULM_DATA_WLABELS_PATH,
    HULM_MODEL_PATH,
    "test-hulm-insert-masks",
    extend_pos_embds=True,
    predict=True,
    preds_split="dev")


if False: # non hulm (control) test run
    evaluate(CONTROL_DATA_WLABELS_PATH,
    CONTROL_MODEL_PATH,
    "test-control-insert-masks",
    extend_pos_embds=False,
    predict=True)

if False: # non hulm (control) dev run
    evaluate(CONTROL_DATA_WLABELS_PATH,
    CONTROL_MODEL_PATH,
    "dev-control-insert-masks",
    extend_pos_embds=False,
    predict=True,
    preds_split="dev")

