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


def loss(probs: torch.Tensor, new_labels: torch.Tensor): # Negative log likelihood
    mask_probs = probs[new_labels != -100]
    loss = -mask_probs.log().mean() # NLL
    return loss

def ppl_from_losses(losses: list): # ppl from a list of loss values
    mean_loss = torch.Tensor(losses).mean()
    return mean_loss.exp().item()

def other_ppl(probs, new_labels): # returns inf seemingly every time
    mask_probs = probs[new_labels != -100]
    n = len(mask_probs)
    ppl = torch.prod(mask_probs)**(-1/n)
    return ppl

def row_losses_users(ds, split=None, return_dict=True): # Returns the CE losses from each row in the split with the user of the row
    losses_list = []
    losses = {}
    split_ds = ds[split] if split is not None else ds
    for row in split_ds:
        probs = torch.Tensor(row['probs'])
        new_labels = torch.Tensor(row['new_labels'])
        if new_labels[new_labels != -100].shape[0] == 0: # no tokens are masked in this row, skip it
            # pdb.set_trace()
            continue
        row_user = row['user_id']
        row_loss = loss(probs, new_labels) # loss function handles removal of unmasked probs
        if math.isnan(row_loss.item()):
            pdb.set_trace()
        losses_list = losses_list + [row_loss]
        if row_user in losses.keys():
            losses[row_user] = losses[row_user] + [row_loss] 
        else:
            losses[row_user] = [row_loss]
    # pdb.set_trace()
    return losses if return_dict else losses_list


def per_doc_ppl(probs_ds, split): # assume already unchunked if hulm
        per_doc_losses = torch.Tensor(row_losses_users(probs_ds, split=split, return_dict=False))
        return per_doc_losses.mean().exp()


def per_user_ppl(probs_ds, split):
    per_doc_losses_dict = row_losses_users(probs_ds, split=split, return_dict=True)
    per_user_ppls = {}
    # first average per user
    for user, user_losses in per_doc_losses_dict.items():
        user_losses = torch.Tensor(user_losses)
        per_user_ppls[user] = user_losses.mean().exp() # doc-level ppl for one user, then average all of those ppls
        # pdb.set_trace()
        # print(per_user_losses[user])
    
    return torch.Tensor(list(per_user_ppls.values())).mean()



def per_split_ppl(ds, split, hulm=None):
    
    all_probs = []
    for m, row_probs in enumerate(ds[split]['probs']):
        all_probs = all_probs + row_probs # this concat takes a long time, but the sum() method was hanging, not sure why

        if m % 500 == 0:
            # print(m)
            pass
    print("new_labels time")

    all_new_labels = []
    for m, row_new_labels in enumerate(ds[split]['new_labels']):
        all_new_labels = all_new_labels + row_new_labels

    all_probs = torch.Tensor(all_probs)
    all_new_labels = torch.Tensor(all_new_labels)
    return ppl_from_losses([loss(all_probs, all_new_labels)])


def df_to_ds(dev_df, test_df):  # combine pandas dfs into splits of a DatasetDict  
    ds = DatasetDict()
    ds['dev'] = Dataset.from_pandas(dev_df, preserve_index=False)
    ds['test'] = Dataset.from_pandas(test_df, preserve_index=False)
    return ds


def print_ppls(probs_ds, ppl_func, hulm=True, suffix = " "):
    blogs_probs, ud_probs = separate_blogs_ud(probs_ds)

    approach = "hulm" if hulm else "traditional"
    approach += suffix
    # print(per_row_ppl(probs_ds, 'dev'))
    print(f"\n{approach}blogs dev ppl: {ppl_func(blogs_probs, 'dev')}")
    print(f"\n{approach}blogs test ppl: {ppl_func(blogs_probs, 'test')}")
    print(f"\n{approach}FB dev ppl: {ppl_func(ud_probs, 'dev')}")
    print(f"\n{approach}FB test ppl: {ppl_func(ud_probs, 'test')}")
    print(f"\n{approach}blogsFB dev ppl: {ppl_func(probs_ds, 'dev')}")
    print(f"\n{approach}blogsFB test ppl: {ppl_func(probs_ds, 'test')}")


    print("\n\n ------- \n\n")
    return

def concat_user_probs(ds):           
    df = ds.to_pandas()
    by_user = df.groupby('user_id', sort=False)
    # np hstack concatenates
    probs_df = by_user['probs'].apply(np.hstack).to_frame().reset_index()
    new_labels_df = by_user['new_labels'].apply(np.hstack).to_frame().reset_index()
    concat_df = probs_df.merge(new_labels_df, on='user_id')
    return concat_df


def unchunk(examples, tokenizer, test_length=-1, wlabels=False):
    eos_token = tokenizer.eos_token_id
    user_ids = []
    text_chunks = []
    attn_mask_chunks = []

    if wlabels:
        probs_chunks = []
        label_chunks = []

    batch_user_ids = examples['user_id']
    batch_input_ids = examples['input_ids']
    batch_attn_masks = examples['attention_mask']
    if wlabels:
        batch_probs = examples['probs']
        batch_new_labels = examples['new_labels']

    for i in range(len(batch_user_ids)):
        user_id = batch_user_ids[i]
        input_ids = batch_input_ids[i]
        attn_mask = batch_attn_masks[i]
        if wlabels:
            probs = batch_probs[i]
            new_labels = batch_new_labels[i]

        if input_ids[len(input_ids)-1] != eos_token:
                print(row, "doesn't end in </s>"); exit()
        
        ends = [0] + [end+1 for end in findall(eos_token, input_ids, len(input_ids)+2)]
        for i in range(len(ends)-1):
            user_ids.append(user_id)
            text_chunks.append(input_ids[ends[i]:ends[i+1]])
            attn_mask_chunks.append(attn_mask[ends[i]:ends[i+1]])
            if wlabels:
                probs_chunks.append(probs[ends[i]:ends[i+1]])
                label_chunks.append(new_labels[ends[i]:ends[i+1]])
    if wlabels:
        return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks,
            'probs' : probs_chunks, 'new_labels' : label_chunks}
    else:
        return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks}


# find all indices in array for a certain element
def findall(n, arr, limit):
    indices = []
    for i in range(min(len(arr), limit+2)):
        if arr[i] == n:
            if arr[i-1] != n or i in {0,1}: # if ==, second /s, doesn't count
                indices.append(i)
    return indices


def unchunk_data(ds, tokenizer, wlabels=False):
    if wlabels:
        unchunked_ds = DatasetDict()
        unchunked_ds['dev'] = ds['dev'].map(lambda row: unchunk(row, tokenizer=tokenizer, wlabels=True, test_length=len(ds['dev'])),
            batched=True, batch_size=1, remove_columns=ds['dev'].column_names)
        unchunked_ds['test'] = ds['test'].map(lambda row: unchunk(row, tokenizer=tokenizer, wlabels=True, test_length=len(ds['test'])),
            batched=True, batch_size=1, remove_columns=ds['test'].column_names)

    else:
        unchunked_ds = ds.map(lambda row: unchunk(row, tokenizer=tokenizer),
            batched=True, batch_size=1000, remove_columns=ds['train'].column_names)

    return unchunked_ds


# breakup single document rows such that they fit into target context size
def breakup(examples, tokenizer, block_size, wlabels=False):
    
    bos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    user_ids = []
    text_chunks = []
    attn_mask_chunks = []
    
    if wlabels:
        probs_chunks = []
        new_labels_chunks = []

    # lists that will make the new dataset at the end
    batch_user_ids = examples['user_id']
    batch_input_ids = examples['input_ids']
    batch_attn_masks = examples['attention_mask']
    if wlabels:
        batch_probs = examples['probs']
        batch_new_labels = examples['new_labels']

    for i in range(len(batch_user_ids)):
        user_id = batch_user_ids[i]
        input_ids = batch_input_ids[i]
        attn_mask = batch_attn_masks[i]

        if wlabels:
            probs = batch_probs[i]
            new_labels = batch_new_labels[i]

        if input_ids[len(input_ids)-1] != eos_token:
            print(i, "doesn't end in </s>"); exit()
        if input_ids[0] not in {tokenizer.bos_token_id, tokenizer.eos_token_id}:
            print(i, "doesn't start with </s> or <s>"); exit()

        input_ids = input_ids[1:len(input_ids)-1] # remove <s> and </s> tokens to be added back later
        attn_mask_start_token, attn_mask_stop_token = attn_mask[0], attn_mask[len(input_ids)-1]
        attn_mask = attn_mask[1:len(attn_mask)-1] # keep attn mask matching input ids

        if wlabels:
            probs_start_token, probs_stop_token = probs[0], probs[len(probs)-1]
            probs = probs[1:len(probs)-1]
            assert len(probs) == len(attn_mask)
            new_labels_start_token, new_labels_stop_token = new_labels[0], new_labels[len(new_labels)-1]
            new_labels = new_labels[1:len(new_labels)-1]
            assert len(new_labels) == len(attn_mask)
            

        adj_block_size = block_size - 2 # account for missing 2 tokens from last two lines

        if len(input_ids) > adj_block_size: # chunk longer blog posts into many pieces
            num = len(input_ids) // adj_block_size # how many full new documents will be made
            token_bounds = [(adj_block_size*(k), adj_block_size*(k+1)) for k in range(num)] + [(adj_block_size*num, len(input_ids))] # e.g. [(0, 510), (510, 1020), (1020, 1500)] for a 1500-token doc and 512 block_size
            attn_mask_bounds = [(adj_block_size*(k), adj_block_size*(k+1)) for k in range(num)] + [(adj_block_size*num, len(attn_mask))] # same bounds for attn mask

            if wlabels:
                probs_bounds = [(adj_block_size*(k), adj_block_size*(k+1)) for k in range(num)] + [(adj_block_size*num, len(probs))] # same bounds for probs
                new_labels_bounds = [(adj_block_size*(k), adj_block_size*(k+1)) for k in range(num)] + [(adj_block_size*num, len(new_labels))] # same bounds for new_labels


            for i in range(len(token_bounds)):
                token_begin, token_end = token_bounds[i]
                attn_mask_begin, attn_mask_end = attn_mask_bounds[i]

                if wlabels:
                    probs_begin, probs_end = probs_bounds[i]
                    new_labels_begin, new_labels_end = new_labels_bounds[i]

                # append
                user_ids.append(user_id)
                text_chunks.append([bos_token] + input_ids[token_begin:token_end] + [eos_token])
                attn_mask_chunks.append([attn_mask_start_token] + attn_mask[attn_mask_begin:attn_mask_end] + [attn_mask_stop_token])
                if wlabels:
                    probs_chunks.append([probs_start_token] + probs[probs_begin:probs_end] + [probs_stop_token])
                    new_labels_chunks.append([new_labels_start_token] + new_labels[new_labels_begin:new_labels_end] + [new_labels_stop_token])


        else: # the whole text for user can be added if it all fits inside block_size
            user_ids.append(user_id)
            text_chunks.append([bos_token] + input_ids + [eos_token])
            attn_mask_chunks.append([attn_mask_start_token] + attn_mask + [attn_mask_stop_token])
            if wlabels:
                probs_chunks.append([probs_start_token] + probs + [probs_stop_token])
                new_labels_chunks.append([new_labels_start_token] + new_labels + [new_labels_stop_token])

    if wlabels:
        return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks,
            'probs' : probs_chunks, 'new_labels' : new_labels_chunks}
    else:
        return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks}


def chunk_data(tokenizer, ds, block_size, one_post=False, wlabels=False):
    if one_post:
        if wlabels:
            chunked_ds = DatasetDict()
            chunked_ds['dev'] = ds['dev'].map(lambda row: breakup(row, tokenizer=tokenizer, block_size=block_size, wlabels=True),
                batched=True, batch_size=1, remove_columns=ds['dev'].column_names)
            chunked_ds['test'] = ds['test'].map(lambda row: breakup(row, tokenizer=tokenizer, block_size=block_size, wlabels=True),
                batched=True, batch_size=1, remove_columns=ds['test'].column_names)

        else:
            chunked_ds = ds.map(lambda rows: breakup(rows, tokenizer=tokenizer, block_size=block_size), batched=True,
                remove_columns=ds['train'].column_names)
        
        return chunked_ds

    assert 1 == 0, "bad chunk_data"


def separate_blogs_ud(ds): # deconstruct ds into (ds, blogs_ds, ud_ds)
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


def describe(df):
    base = df.describe()
    base.loc["1%"] = df.quantile(0.01)
    base.loc["99%"] = df.quantile(0.99)
    return base.reindex(["count", "mean", "std", "min", "1%", "25%", "50%", "75%", "99%", "max"])


def describe_lens(ds, split='train', user=False):
    df = ds[split].to_pandas()

    print("total tokens: ", df['input_ids'].str.len().sum())

    if user:
        print("Not implemented yet")
    else:
        print(describe(df['input_ids'].str.len()))

    

# method to help pick a free GPU
def pick_gpu():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    for j in range(len(memory_free_values)):
        if memory_free_values[j] == 48676:
            print(f"using GPU {j}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(j)
            return

    print("no GPU available"); exit()

pick_gpu()


base_data_path = "/cronus_data/ssmith/data/blogsUD/"
control_probs_path = base_data_path + "control_probs_ds"
hulm_probs_path = base_data_path + "hulm_probs_ds"
docss_path = base_data_path + "_extra_probs_dses/chunked_docss_4096_WLABELS"
unchunked_path = base_data_path + "_extra_probs_dses/unchunked_4096_TESTLABELS"
MAY_control_probs_path = base_data_path + "MAY_control_probs_ds"
OOTB_probs_path = base_data_path + "OOTB_probs_ds"

hulm_probs_ds = load_from_disk(hulm_probs_path)
docss = load_from_disk(docss_path)
control_probs_ds = load_from_disk(control_probs_path)
# MAY_control_probs_ds = load_from_disk(MAY_control_probs_path)
OOTB_probs_ds = load_from_disk(OOTB_probs_path)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")


 


if True: # OOTB  ppls
    print_ppls(OOTB_probs_ds, per_user_ppl, hulm=False, suffix=" per user ")
    print_ppls(OOTB_probs_ds, per_doc_ppl, hulm=False, suffix=" per doc ")
    exit()



if False: # MAY control ppls
    print_ppls(MAY_control_probs_ds, per_user_ppl, hulm=False, suffix=" per user ")
    print_ppls(MAY_control_probs_ds, per_doc_ppl, hulm=False, suffix=" per doc ")



if False:
    unchunked_4096_hulm_probs_ds = unchunk_data(hulm_probs_ds, tokenizer, wlabels=True)
    unchunked_512_hulm_probs_ds = chunk_data(tokenizer, unchunked_4096_hulm_probs_ds, 512, one_post=True, wlabels=True)

    print_ppls(unchunked_512_hulm_probs_ds, per_user_ppl, hulm=True, suffix=" per user ")
    print_ppls(control_probs_ds, per_user_ppl, hulm=False, suffix=" per user ")


if True:
    unchunked_4096_hulm_probs_ds = unchunk_data(hulm_probs_ds, tokenizer, wlabels=True)
    unchunked_512_hulm_probs_ds = chunk_data(tokenizer, unchunked_4096_hulm_probs_ds, 512, one_post=True, wlabels=True)

    print_ppls(unchunked_512_hulm_probs_ds, per_doc_ppl, hulm=True, suffix=" per doc ")
    print_ppls(control_probs_ds, per_doc_ppl, hulm=False, suffix=" per doc ")



if False: # 
    print_ppls(control_probs_ds, per_user_ppl, hulm=False, suffix=" per user ")
    # print_ppls(hulm_probs_ds, per_user_ppl, hulm=True, suffix=" per user ")


if False: # per split ppls
    print_ppls(control_probs_ds, per_split_ppl, hulm=False, suffix=" per split ")
    print_ppls(hulm_probs_ds, per_split_ppl, hulm=True, suffix=" per split ")




if False: # hulm dev and test ppl on whole dataset
    dev_ppl = per_split_ppl(hulm_probs_ds, "dev", " per user ")
    test_ppl = per_split_ppl(hulm_probs_ds, "test", " per user ")
    print(dev_ppl, test_ppl)



# print(f"{i}: {ppl(probs, new_labels)}")




