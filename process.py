from datasets import load_from_disk, concatenate_datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from random import sample
import re
import time


def tokenize_data(tokenizer, untokenized_path=None):
    dataset = load_from_disk(untokenized_path)
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=False)
    print("start tokenizing data...")
    tokenized_data = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])
    print("data tokenized.")
    tokenized_data.save_to_disk("/cronus_data/ssmith/data/blogs_ud/tokenized_corpus")
    return tokenized_data


def concat_df(ds): # already split            
    df = ds.to_pandas()
    by_user = df.groupby('user_id', sort=False)
    # np hstack concatenates
    tokens_df = by_user['input_ids'].apply(np.hstack).to_frame().reset_index()
    attn_mask_df = by_user['attention_mask'].apply(np.hstack).to_frame().reset_index()
    concat_df = tokens_df.merge(attn_mask_df, on='user_id')
    return concat_df


def remove_verbose_users(ds, p=0.99):
    df = ds.to_pandas()
    user_df = df.groupby('user_id', sort=False)

    non_verbose_user_df = user_df['input_ids'].apply(lambda x: x.str.len().median())
    med_length_cutoff = non_verbose_user_df.quantile(p)
    non_verbose_users = non_verbose_user_df[non_verbose_user_df < med_length_cutoff].index.tolist()
    non_verbose_df = df[ df['user_id'].isin(non_verbose_users)]
    return non_verbose_df 
    

def remove_short_docs(ds, min_block_size=500):
    # remove rows with input_ids shorter than min_block_size
    print("unfiltered train length ", len(ds['train']))
    ds = ds.filter(lambda row: len(row['input_ids']) >= min_block_size)
    print("filtered train length: ", len(ds['train']))
    return ds


def remove_long_docs(ds, max_block_size):
    # remove docs greater than max_block_size
    print("unfiltered train length ", len(ds['train']))
    ds = ds.filter(lambda row: len(row['input_ids']) <= max_block_size)
    print("filtered train length: ", len(ds['train']))


# find all indices in array for a certain element
def findall(n, arr, limit):
    indices = []
    for i in range(min(len(arr), limit+2)):
        if arr[i] == n:
            indices.append(i)
    return indices


# cutoff user posts at nearest document end before block_size
def chunk_one_row(examples, tokenizer, block_size):
    
    eos_token = tokenizer.eos_token_id
    user_ids = []
    text_chunks = []
    attn_mask_chunks = []

    batch_user_ids = examples['user_id']
    batch_input_ids = examples['input_ids']
    batch_attn_masks = examples['attention_mask']

    for i in range(len(batch_user_ids)):
        user_id = batch_user_ids[i]
        input_ids = batch_input_ids[i]
        attn_mask = batch_attn_masks[i]

        if input_ids[len(input_ids)-1] != eos_token:
                print(row, "doesn't end in </s>"); exit()
        
        if len(input_ids) >= block_size:
            ends = [end+1 for end in findall(eos_token, input_ids, block_size)]
            cutoff = 0
            for end in ends:
                if end <= block_size:
                    cutoff = end
                else:
                    break
            # cutoff = 0 means that the first doc was over block_size, user doesn't included at all
            if cutoff > 0:
                user_ids.append(user_id)
                text_chunks.append(input_ids[:cutoff])
                attn_mask_chunks.append(attn_mask[:cutoff])
            
        else: # all user text fits within block_size
            user_ids.append(user_id)
            text_chunks.append(input_ids)
            attn_mask_chunks.append(attn_mask)

    return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks}


def truncate(examples, tokenizer, block_size):
    eos_token = tokenizer.eos_token_id
    bos_token = tokenizer.bos_token_id
    batch_user_ids = examples['user_id']
    batch_input_ids = examples['input_ids']
    batch_attn_masks = examples['attention_mask']
    for i in range(len(batch_user_ids)):
        trunc = batch_input_ids[i][:block_size-1]
        if trunc[len(trunc)-1] != eos_token:
            trunc.append(eos_token)
        else: # last token is </s>
            if trunc[len(trunc)-2] == eos_token: # should only be one </s> token ending a row
                del trunc[len(trunc)-1]
        batch_input_ids[i] = trunc
        batch_attn_masks[i] = batch_attn_masks[i][:len(trunc)]
    return {'user_id' : batch_user_ids,
            'input_ids' : batch_input_ids,
            'attention_mask' : batch_attn_masks}


# make multiple rows per user such that none exceed block_size
def chunk_multiple_rows(examples, tokenizer, block_size):
    
    eos_token = tokenizer.eos_token_id
    user_ids = []
    text_chunks = []
    attn_mask_chunks = []

    batch_user_ids = examples['user_id']
    batch_input_ids = examples['input_ids']
    batch_attn_masks = examples['attention_mask']

    for i in range(len(batch_user_ids)):
        user_id = batch_user_ids[i]
        input_ids = batch_input_ids[i]
        attn_mask = batch_attn_masks[i]

        if input_ids[len(input_ids)-1] != eos_token:
            print(row, "doesn't end in </s>"); exit()

        if len(input_ids) >= block_size: # chunk longer blog posts into many pieces
            begin = 0
            last_doc_end = 0

            for idx, token in enumerate(input_ids):
                append = False

                if token == eos_token:
                    last_doc_end = idx # track last sep token seen
                    if idx == len(input_ids)-1:
                        append = True # make sure last document gets included

                if idx - begin == block_size-1: # max context size reached, chunk needs to be made
                    append = True
                
                if append: # conditions have been met to add a chunk
                    user_ids.append(user_id)

                    if last_doc_end == -1:
                        print(input_ids, idx, begin, "Document greater than block_size present"); exit()
                    if last_doc_end - begin > block_size:
                        print(input_ids, idx, begin, "Chunk bigger than block size"); exit()

                    text_chunks.append(input_ids[begin:last_doc_end+1])
                    attn_mask_chunks.append(attn_mask[begin:last_doc_end+1])

                    begin = last_doc_end+1 # start off after last chunked document
                    last_doc_end = -1 # should always get re-updated before next block size
        
        else: # the whole text for user can be added if it all fits inside block_size
            user_ids.append(user_id)
            text_chunks.append(input_ids)
            attn_mask_chunks.append(attn_mask)    
    return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks}


def chunk_data(tokenizer, ds, block_size, prune=False, multiple_rows=False):
    if prune == True and multiple_rows == True:
        print("Not implemented yet"); exit()
    if multiple_rows: # multiple rows per user
        chunked_ds = ds.map(lambda rows: chunk_mutiple_rows(rows, tokenizer=tokenizer,
            block_size=block_size), batched=True,
            remove_columns=ds['train'].column_names) #, num_proc=4)
    else: # one row per user
        if prune:
            chunked_ds = ds.map(lambda row: truncate(row, tokenizer=tokenizer,
                block_size=block_size), batched=True,
                remove_columns=ds['train'].column_names)
        else:
            chunked_ds = ds.map(lambda row: chunk_one_row(row, tokenizer=tokenizer,
                block_size=block_size), batched=True,
                remove_columns=ds['train'].column_names)

    return chunked_ds


def unchunk(examples, tokenizer):
    eos_token = tokenizer.eos_token_id
    user_ids = []
    text_chunks = []
    attn_mask_chunks = []

    batch_user_ids = examples['user_id']
    batch_input_ids = examples['input_ids']
    batch_attn_masks = examples['attention_mask']

    for i in range(len(batch_user_ids)):
        user_id = batch_user_ids[i]
        input_ids = batch_input_ids[i]
        attn_mask = batch_attn_masks[i]

        if input_ids[len(input_ids)-1] != eos_token:
                print(row, "doesn't end in </s>"); exit()
        
        ends = [0] + [end+1 for end in findall(eos_token, input_ids, len(input_ids)+2)]
        for i in range(len(ends)-1):
            user_ids.append(user_id)
            text_chunks.append(input_ids[ends[i]:ends[i+1]])
            attn_mask_chunks.append(attn_mask[ends[i]:ends[i+1]])
    return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks}


def unchunk_data(tokenizer, ds):
    unchunked_ds = ds.map(lambda row: unchunk(row, tokenizer=tokenizer),
        batched=True, remove_columns=ds['train'].column_names)
    return unchunked_ds


def df_to_ds(train_df, val_df, test_df=None):    
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    ds = DatasetDict()
    ds['train'] = train_ds
    ds['validation'] = val_ds
    if test_df is not None:
        ds['test'] = Dataset.from_pandas(test_df, preserve_index=False)
    return ds


def sample_users(ds, p=0.05):
    user_ids = ds['user_id']
    assert p > 0 and p < 1, "proportion must be between 0 and 1"
    user_sample = sample(user_ids, int(len(user_ids)*p))
    small_ds = ds.filter(lambda row: row['user_id'] in user_sample)
    return small_ds


def sample_decode(ds, n=5, split='train'):
    df = ds[split].to_pandas()
    for i in range(n):
        row = df.iloc[i]
        time.sleep(3)
        print(row[['user_id']], tokenizer.decode(row['input_ids']))
    print("\n\n\n\n")


def fixed_date_all_df(ds):
    print(ds)
    splits = [ds[key] for key in ds.keys()]
    all_ds = splits[0]
    for i in range(len(splits)-1):
        assert splits[i].features.type == splits[i+1].features.type
        all_ds = concatenate_datasets([all_ds, splits[i+1]])
    print(all_ds)
    df = all_ds.to_pandas()
  
    if True: # fix months in wrong languages in the dates
        du_en = {'januari': 'january', 'februari': 'february', 'maart': 'march',
                'april': 'april', 'mei': 'may', ',juni,': ',june,', ',juli,': ',july,',
                'augustus': 'august', 'september': 'september', 'oktober': 'october',
                'november': 'november', 'december': 'december'}
        es_en = {'Jaanuar': 'january', 'Veebruar': 'february', 'Märts': 'march',
                ',Aprill,': ',april,', 'Mai': 'may', 'Juuni': 'June', 'Juuli': 'July',
                'August': 'august', 'September': 'september', 'Oktoober': 'october',
                'November': 'november', 'Detsember': 'december'}
        fn_en = {'tammikuu': 'january', 'helmikuu': 'february', 'maaliskuu': 'march',
                'huhtikuu': 'april', 'toukokuu': 'may', 'kesäkuu': 'june', 'heinäkuu': 'july',
                'elokuu': 'august', 'syyskuu': 'september',
                'lokakuu': 'october', 'marraskuu': 'november', 'joulukuu': 'december'}
        it_en = {'gennaio': 'january', 'febbraio': 'february', 'marzo': 'march',
                'aprile': 'april', 'maggio': 'may', 'giugno': 'june', 'luglio': 'july',
                'agosto': 'august', 'settembre': 'september', 'ottobre': 'october',
                'novembre': 'november', 'dicembre': 'december'}  
        gr_en_up = {",Januar," : ",january,", ",Februar," : ",february,", ",Marz," : ",march,",
                ",April," : ",april,", ",Mai," : ",maiy,", ",Juni,": ",june,", ",Juli,": ",july,",
                ",August,": ",august,", ",September,": ",september,", ",Oktober,": ",october,",
                ",November,": ",november,", ",Dezember,": ",december,"}
        gr_en = {",januar," : ",january,", ",februar," : ",february,", ",marz," : ",march,",
                ",april," : ",april,", ",mai," : ",may,", ",juni,": ",june,", ",juli,": ",july,",
                ",august,": ",august,", ",september,": ",september,", ",oktober,": ",october,",
                ",november,": ",november,", ",dezember,": ",december,"}
        po_en = {"Janeiro" : "january", "Fevereiro" : "february", "Marco" : "march", "Abril" : "april",
                "Maio" : "may", "Junho": "june", "Julho": "july", "Agosto": "august",
                "Setembro": "september", "Outubro": "october", "Novembro": "november",
                "Dezembro": "december"}
        sp_en = {"enero" : "january", "febrero" : "february", "marzo" : "march", "abril" : "april",
                "mayo" : "may", "junio": "june", "julio": "july", "agosto": "august",
                "septiembre": "september", "octubre": "october", "noviembre": "november",
                "diciembre": "december"}
        fr_en = {"janvier" : "january", "fevrier" : "february", "mars" : "march", "avril" : "april",
                "mai" : "may", ",juin,": ",june,", "juillet": "july", "aout": "august",
                "septembre": "september", "octobre": "october", "novembre": "november",
                "decembre": "december"}
        bad_months = [du_en, es_en, fn_en, it_en, gr_en, gr_en_up, po_en, sp_en, fr_en]
        for month_dict in bad_months:
            for wrong, en in month_dict.items():
                df['date'] = df['date'].str.replace(wrong, en)
        
    month_pattern = r'(?:,(?!january,|february,|march,|april,|may,|june,|july,|august,|september,|october,|november,|december,).*,)'
    bad_month_df = df[df['date'].str.contains(month_pattern, case=False, regex=True)]
    bad_month_users = bad_month_df['user_id'].unique()
    df = df[~df['user_id'].isin(bad_month_users)] # remove users whose date is incorrectly formatted
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['user_id', 'date'], ascending=[True, False])
    for i in range(5):
        print(df[['user_id', 'date']][1000*i:1000*i+15])
    all_ds = Dataset.from_pandas(df, preserve_index=False)
    print(all_ds)
    return all_ds


def reset_splits(all_ds, p_train=0.9, p_val=0.05):

    all_users = list(set(all_ds['user_id']))
    non_train_users = sample(all_users, int(len(all_users) * (1-p_train)))
    p_val = p_val / (1-p_train)
    n = int(len(non_train_users) * p_val)
    val_users = non_train_users[:n]
    test_users = non_train_users[n:]

    new_ds = DatasetDict()
    new_ds['train'] = all_ds.filter(lambda row: row['user_id'] not in non_train_users)
    new_ds['validation'] = all_ds.filter(lambda row: row['user_id'] in val_users)
    new_ds['test'] = all_ds.filter(lambda row: row['user_id'] in test_users)
    return new_ds


def remove_extra_cls(example, tokenizer):
    input_ids = example['input_ids']
    cls_token = tokenizer.bos_token_id 
    assert input_ids[0] == cls_token
    necls_input_ids = [input_ids[0]] + [t for t in input_ids[1:] if t != cls_token]
    example['input_ids'] = necls_input_ids
    example['attention_mask'] = example['attention_mask'][:len(necls_input_ids)]
    return example


def remove_extra_cls_data(ds, tokenizer):
    ds = ds.map(lambda row: remove_extra_cls(row, tokenizer=tokenizer), num_proc=4)
    return ds


def _add_sep(example, tokenizer):
    input_ids = example['input_ids']
    attn_mask = example['attention_mask']
    sep_token = tokenizer.eos_token_id # roberta tokenizer shares a sep and eos token
    dsep_input_ids = []
    dsep_attn_mask = []

    for i in range(len(input_ids)):
        dsep_input_ids.append(input_ids[i])
        dsep_attn_mask.append(attn_mask[i])
        if input_ids[i] == sep_token and i != len(input_ids)-1:
            dsep_input_ids.append(sep_token)
            dsep_attn_mask.append(attn_mask[i])
    example['input_ids'] = dsep_input_ids
    example['attention_mask'] = dsep_attn_mask
    return example


def add_sep(ds, tokenizer):
    ds = ds.map(lambda row: _add_sep(row, tokenizer=tokenizer), num_proc=4)
    return ds


def describe(df):
    base = df.describe()
    base.loc["1%"] = df.quantile(0.01)
    base.loc["99%"] = df.quantile(0.99)
    return base.reindex(["count", "mean", "std", "min", "1%", "25%", "50%", "75%", "99%", "max"])


def describe_lens(ds, user=False, minmax=False):
    # print(tokenized_user_blog_corpus)   
    df = ds['train'].to_pandas()

    # print token sum
    print("total tokens: ", df['input_ids'].str.len().sum())

    if user:
        user_df = df.groupby('user_id', sort=False)
        print(f"\n---- User median post length ---\n {describe(user_df['input_ids'].apply(lambda x: x.str.len().median()))}")
        if minmax:
            print(f"\n---- User min post length ---\n {describe(user_df['input_ids'].apply(lambda x: x.str.len().min()))}")
            print(f"\n---- User max post length ---\n {describe(user_df['input_ids'].apply(lambda x: x.str.len().max()))}")
    else:
        print(describe(df['input_ids'].str.len()))


# ----- MAIN -----

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
block_size = 4096
multiple_rows = False

base_path = "/cronus_data/ssmith/data/blogsUD/"
untokenized_path = base_path + "blogsUD_corpus"
tokenized_path = base_path + "tokenized_corpus"
sorted_all_path = base_path + "sorted_all_ds"
split_path = base_path + "split_ds"
nv_path = base_path + "split_nv"
concat_path = base_path + "concat_ds"
concat_dsep_path = base_path + "concat_dsep"
sample_concat_path = base_path + "concat_sample"
sample_concat_dsep_path = base_path + "concat_sample_dsep"
chunked_path = base_path + f"chunked_{str(block_size)}"
sample_chunked_path = base_path + f"sample_chunked_{str(block_size)}"
sample_chunked_dsep_path = base_path + f"sample_chunked_dsep_{str(block_size)}"
unchunked_path = base_path + f"unchunked_{str(block_size)}"


if multiple_rows:
    chunked_path += "_mr"


if True:
    concat_ds = load_from_disk(concat_path)
    describe_lens(concat_ds)
    exit()


if False:
    sample_chunked = load_from_disk(sample_chunked_path)
    sample_chunked_dsep = load_from_disk(sample_chunked_dsep_path)
    describe_lens(sample_chunked)
    describe_lens(sample_chunked_dsep)
    exit()

if False: # chunk both and remove short docs on both and save
    sample_concat = load_from_disk(sample_concat_path)
    sample_concat_dsep = load_from_disk(sample_concat_dsep_path)
    sample_chunked = chunk_data(tokenizer, sample_concat, block_size, prune=True)
    sample_chunked = remove_short_docs(sample_chunked, 500)
    sample_chunked_dsep = chunk_data(tokenizer, sample_concat_dsep, block_size, prune=True)
    sample_chunked_dsep = remove_short_docs(sample_chunked_dsep, 500)
    describe_lens(sample_chunked)
    describe_lens(sample_chunked_dsep)
    sample_chunked.save_to_disk(sample_chunked_path)
    sample_chunked_dsep.save_to_disk(sample_chunked_dsep_path) 


if False: # create dsep from concat sample
    ds = load_from_disk(sample_concat_path)
    describe_lens(ds)
    necls_ds = remove_extra_cls_data(ds, tokenizer)
    describe_lens(necls_ds)
    dsep_ds = add_sep(necls_ds, tokenizer)
    describe_lens(dsep_ds)
    dsep_ds.save_to_disk(sample_concat_dsep_path)
    exit()


if False: # sample from concat
    ds = load_from_disk(concat_path)
    describe_lens(ds)
    sample_train = sample_users(ds['train'])
    describe_lens(ds)
    sample_ds = DatasetDict()
    sample_ds['train'] = sample_train
    for key in ds.keys():
        if key != 'train':
            sample_ds[key] = ds[key]
    describe_lens(sample_ds)
    sample_ds.save_to_disk(sample_concat_path)


if False: # split to concat
    ds = load_from_disk(split_path)
    print("before nv:")
    describe_lens(ds, user=True) 
    nv_train_df = remove_verbose_users(ds['train'])
    nv_val_df = remove_verbose_users(ds['validation'])
    nv_test_df = remove_verbose_users(ds['test'])
    nv_ds = df_to_ds(nv_train_df, nv_val_df, nv_test_df)
    print("\nafter nv:")
    describe_lens(nv_ds, user=True)
    concat_train_df = concat_df(nv_ds['train']) 
    concat_val_df = concat_df(nv_ds['validation'])
    concat_test_df = concat_df(nv_ds['test'])
    concat_ds = df_to_ds(concat_train_df, concat_val_df, concat_test_df)
    print("\nafter concat:")
    describe_lens(concat_ds)
    concat_ds.save_to_disk(concat_path); exit()


if False:
    chunked_nvns = load_from_disk(chunked_nvns_path)
    dsep_chunked = add_sep_data(chunked_nvns, tokenizer)
    exit()

if False: # unchunk non-verbose no small 4096 chunked data
    chunked_nvns = load_from_disk(chunked_nvns_path)
    describe_lens(chunked_nvns)
    unchunked_ds = unchunk_data(tokenizer, chunked_nvns)
    describe_lens(unchunked_ds)
    unchunked_ds.save_to_disk(unchunked_nvns_path)


if False: # concat nv 
    nv_ds = load_from_disk(nv_path)
    describe_lens(nv_ds, user=True)
    concat_train_df = concat_df(nv_ds['train']) 
    concat_val_df = concat_df(nv_ds['validation'])
    concat_test_df = concat_df(nv_ds['test'])
    concat_nv_ds = df_to_ds(concat_train_df, concat_val_df, concat_test_df)
    describe_lens(concat_nv_ds)
    concat_nv_ds.save_to_disk(concat_nv_path)


if False: # chunk and remove under 500 tokens from nv concat
    concat_nv = load_from_disk(concat_nv_path)
    chunked_nv = chunk_data(tokenizer, concat_nv, block_size, multiple_rows)
    describe_lens(chunked_nv)
    chunked_nvns = remove_short_docs(chunked_nv, 500)
    describe_lens(chunked_nvns)
    chunked_nvns.save_to_disk(chunked_nvns_path)



if False:
    ds = load_from_disk(sorted_all_path)
    ds = reset_splits(ds)
    print(ds)
    ds.save_to_disk(split_path)


if False:
    ds = load_from_disk(split_path)
    nv_train_df = remove_verbose_users(ds['train'])
    nv_val_df = remove_verbose_users(ds['validation'])
    nv_test_df = remove_verbose_users(ds['test'])
    nv_ds = df_to_ds(nv_train_df, nv_val_df, nv_test_df)
    describe_lens(nv_ds)
    nv_ds.save_to_disk(nv_path)
    

if False:
    ds = load_from_disk(tokenized_path)
    sorted_all_ds = fixed_date_all_df(ds)
    sorted_all_ds.save_to_disk(sorted_all_path)

if False:
    ds = load_from_disk(tokenized_path)
    train_df = ds['train'].to_pandas()    
    val_df = ds['validation'].to_pandas()
    train_df['user_id'] = train_df['user_id'].apply(int)
    print(len(train_df[train_df['user_id'] > 5000]))

    user_ids = ds['train']['user_id']
    print(list(set(user_ids).intersection(ds['validation']['user_id'])))


if False:
    chunked_nvns = load_from_disk(chunked_nv_path)
    user_ids = chunked_nvns['train']['user_id']
    print(len(set(user_ids).intersection(set(chunked_nvns['validation']['user_id'])))); exit()
    print(chunked_nvns)
    chunked_nvns = add_test_split(chunked_nvns)
    print(chunked_nvns)


if False: # analyze some responses from chunked vs unchunked
    chunked_nvns = load_from_disk(chunked_nv_path)
    unchunked_nvns = load_from_disk(unchunked_nvns_path)
    for i in range(15):
        print(tokenizer.decode(unchunked_nvns['train'][i]['input_ids']), unchunked_nvns['train'][i]['user_id'], "\n\n")

    print("\n\n\n\n\n\n")
    for i in range(2):
        print(tokenizer.decode(chunked_nvns['train'][i]['input_ids']), chunked_nvns['train'][i]['user_id'], "\n\n")

           



if False: # create non verbose set
    tokenized_ds = load_from_disk(tokenized_path)
    describe_lens(tokenized_ds, user=True)
    nv_train_df = remove_verbose_users(tokenized_ds['train'])
    nv_val_df = remove_verbose_users(tokenized_ds['validation'])
    nv_ds = df_to_ds(nv_train_df, nv_val_df)
    nv_ds.save_to_disk(nv_path)

if False:
    chunked_ds = load_from_disk(chunked_path)
    unchunked_ds = unchunk_data(tokenizer, chunked_ds)
    describe_lens(unchunked_ds)
    unchunked_ds.save_to_disk(unchunked_path)

if False: # chunk necls and normal sample
    small_concat_ds = load_from_disk(small_concat_path)
    small_concat_necls = load_from_disk(small_concat_necls_path)

    small_chunked_ds = chunk_data(tokenizer, small_concat_ds, block_size, multiple_rows)
    small_chunked_necls = chunk_data(tokenizer, small_concat_necls, block_size, multiple_rows)
    small_chunked_ds.save_to_disk(small_chunked_path)
    small_chunked_necls.save_to_disk(small_chunked_necls_path)


if False: # create necls set from concat sample
    small_concat_ds = load_from_disk(small_concat_path)
    small_concat_necls = remove_extra_cls_data(small_concat_ds, tokenizer)
    small_concat_necls.save_to_disk(small_concat_necls_path)
    

if False: # create 5% user sample of concatenated data
    concat_ds = load_from_disk(concat_ds_path)
    print(concat_ds)
    small_train = sample_users(concat_ds['train'])
    print(small_train)
    small_ds = DatasetDict()
    small_ds['train'] = small_train
    small_ds['validation'] = concat_ds['validation'] # no need to sample for val set
    small_ds.save_to_disk(small_concat_path)


if False: # chunk necls concat data
    concat_ds_necls = load_from_disk(concat_necls_path)
    describe_lens(concat_ds_necls)
    chunked_ds_necls = chunk_data(tokenizer, concat_ds_necls, block_size, multiple_rows)
    describe_lens(chunked_ds_necls)
    chunked_ds = load_from_disk(chunked_path)
    describe_lens(chunked_ds)

if False: # sample 5% of users from chunked ds and save
    chunked_ds = load_from_disk(chunked_path)
    small_train = sample_users(chunked_ds['train'])
    small_val = sample_users(chunked_ds['validation'])

    small_ds = DatasetDict()
    small_ds['train'] = small_train
    small_ds['validation'] = chunked_ds['validation'] # no need to sample for val set

    small_ds.save_to_disk(small_chunked_path)

if False: # remove extra cls from chunked data
    small_ds = load_from_disk(small_chunked_path)
    describe_lens(small_ds['train'].to_pandas())
    # necls = no extra cls
    small_ds_necls = remove_extra_cls_data(small_ds, tokenizer)
    print('tokens removed')
    describe_lens(small_ds_necls['train'].to_pandas())



if False: # remove extra cls from concat and save
    train_concat_df = pd.read_pickle(train_concat_path)
    val_concat_df = pd.read_pickle(val_concat_path)
    concat_ds = df_to_ds(train_concat_df, val_concat_df)
    concat_ds.save_to_disk(concat_ds_path)
    concat_ds_necls = remove_extra_cls_data(concat_ds, tokenizer)
    describe_lens(concat_ds_necls['train'].to_pandas())
    concat_ds_necls.save_to_disk(concat_necls_path)

if False: # no tokenize, no filter, concat, chunk one row
    tokenized_ds = load_from_disk(tokenized_path)
    describe_lens(tokenized_ds['train'].to_pandas(), user=True)

    train_concat_df = concat_df('train', tokenized_ds['train'], concat_path)
    val_concat_df = concat_df('validation', tokenized_ds['validation'], concat_path)

    describe_lens(train_concat_df, user=False)
    chunked_ds = chunk_data(tokenizer, train_concat_df, val_concat_df, block_size, multiple_rows)

    print("data chunked, saving data...")
    chunked_ds.save_to_disk(chunked_path)
    print(f"data saved for block size {block_size}")


if False:

    if not tokenized:
        untokenized_ds = load_from_disk(untokenized_path)
        tokenized_ds = tokenize_data(tokenizer, untokenized_ds, tokenized_path)
    elif not filtered:
        tokenized_ds = load_from_disk(tokenized_path)
        filtered_ds = filter_ds(tokenized_ds, block_size, filtered_path)
    elif not concat:
        filtered_ds = load_from_disk(filtered_path)
        # concatenates and changes to pandas df
        train_concat_df = get_concat_df(block_size, filtered_ds['train'], train_concat_path)
        val_concat_df = get_concat_df(block_size, filtered_ds['validation'], val_concat_path)
    else:
        train_concat_df = pd.read_pickle(concat_path + 'train')
        val_concat_df = pd.read_pickle(concat_path + 'validation')


    # chunks and changes from pandas dfs to one HF DatasetDict
    describe_lens(train_concat_df)
    chunked_ds = chunk_data(tokenizer, train_concat_df, val_concat_df, block_size, multiple_rows)


