from datasets import load_from_disk, concatenate_datasets
from datasets import Dataset, DatasetDict, Features, Value
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
                        print(tokenizer.decode(input_ids), idx, begin, "Document greater than block_size present"); exit()
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


# breakup single document rows such that they fit into target context size
def breakup(examples, tokenizer, block_size):
    
    bos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    user_ids = []
    text_chunks = []
    attn_mask_chunks = []

    # lists that will make the new dataset at the end
    batch_user_ids = examples['user_id']
    batch_input_ids = examples['input_ids']
    batch_attn_masks = examples['attention_mask']

    for i in range(len(batch_user_ids)):
        user_id = batch_user_ids[i]
        input_ids = batch_input_ids[i]
        attn_mask = batch_attn_masks[i]

        if input_ids[len(input_ids)-1] != eos_token:
            print(row, "doesn't end in </s>"); exit()
        if input_ids[0] != tokenizer.bos_token_id:
            print(row, "doesn't start with <s>"); exit()

        input_ids = input_ids[1:len(input_ids)-1] # remove <s> and </s> tokens to be added back later
        attn_mask_start_token, attn_mask_stop_token = attn_mask[0], attn_mask[len(input_ids)-1]
        attn_mask = attn_mask[1:len(attn_mask)-1] # keep attn mask matching input ids
        adj_block_size = block_size - 2 # account for missing 2 tokens from last two lines

        if len(input_ids) > adj_block_size: # chunk longer blog posts into many pieces
            num = len(input_ids) // adj_block_size # how many full new documents will be made
            token_bounds = [(adj_block_size*(k), adj_block_size*(k+1)) for k in range(num)] + [(adj_block_size*num, len(input_ids))] # e.g. [(0, 510), (510, 1020), (1020, 1500)] for a 1500-token doc and 512 block_size
            attn_mask_bounds = [(adj_block_size*(k), adj_block_size*(k+1)) for k in range(num)] + [(adj_block_size*num, len(attn_mask))] # same bounds for attn mask
            for i in range(len(token_bounds)):
                token_begin, token_end = token_bounds[i]
                attn_mask_begin, attn_mask_end = attn_mask_bounds[i]

                if user_id == "1171643" and False: # test user
                    print(f"{i}th row: appending for ({token_begin}, {token_end})")
                    print(tokenizer.decode([bos_token] + input_ids[token_begin:token_end] + [eos_token]))

                # append
                user_ids.append(user_id)
                text_chunks.append([bos_token] + input_ids[token_begin:token_end] + [eos_token])
                attn_mask_chunks.append([attn_mask_start_token] + attn_mask[attn_mask_begin:attn_mask_end] + [attn_mask_stop_token])
        else: # the whole text for user can be added if it all fits inside block_size
            user_ids.append(user_id)
            text_chunks.append([bos_token] + input_ids + [eos_token])
            attn_mask_chunks.append([attn_mask_start_token] + attn_mask + [attn_mask_stop_token])    
    return {'user_id' : user_ids, 'input_ids' : text_chunks, 'attention_mask' : attn_mask_chunks}


def chunk_data(tokenizer, ds, block_size, prune=False, multiple_rows=False, one_post=False):
    if prune == True and multiple_rows == True:
        print("Not implemented yet"); exit()
    if one_post:
        chunked_ds = ds.map(lambda rows: breakup(rows, tokenizer=tokenizer,
            block_size=block_size), batched=True,
            remove_columns=ds['train'].column_names) #, num_proc=4)
        return chunked_ds
    
    if multiple_rows: # multiple rows per user
        chunked_ds = ds.map(lambda rows: chunk_multiple_rows(rows, tokenizer=tokenizer,
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


def unchunk_data(ds, tokenizer):
    unchunked_ds = ds.map(lambda row: unchunk(row, tokenizer=tokenizer),
        batched=True, remove_columns=ds['train'].column_names)
    return unchunked_ds



def df_to_ds(train_df, dev_df, test_df=None):    
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    dev_ds = Dataset.from_pandas(dev_df, preserve_index=False)
    ds = DatasetDict()
    ds['train'] = train_ds
    ds['dev'] = dev_ds
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
    print(blogs_ds, ud_ds)
    return blogs_ds, ud_ds


def sample_users(ds, p=0.05):
    user_ids = ds['user_id']
    assert p > 0 and p < 1, "proportion must be between 0 and 1"
    user_sample = sample(user_ids, int(len(user_ids)*p))
    small_ds = ds.filter(lambda row: row['user_id'] in user_sample)
    return small_ds


def sample_users_from_other_ds(ds, ref_ds):
    target_user_ids = list(set(ref_ds['user_id']))
    df = ds.to_pandas()
    sample_ds = Dataset.from_pandas(df[df['user_id'].isin(target_user_ids)], preserve_index=False)
    return sample_ds


def sample_decode(ds, n=5, split='train',index_list=[]):
    df = ds[split].to_pandas()
    if index_list == []:
        index_list = np.arange(n)
    for i in index_list:
        row = df.iloc[i]
        time.sleep(3)
        print(row[['user_id']], tokenizer.decode(row['input_ids']))
    print("\n\n\n\n")


def decode_user(ds, from_index=True, user_id=4178076, user_index=803, split='train', n=5):
    df = ds[split].to_pandas()
    if from_index:
        user_id = df['user_id'].unique()[user_index]
    print(f"---- USER {user_id} ----")
    print(type(df['user_id'].iloc[0]))
    user_df = df[df['user_id'] == str(user_id)]
    if len(user_df) == 0:
        print(f"user {user_id} not found")
    for i in range(min(len(user_df), n)):
        row = user_df.iloc[i]
        print(tokenizer.decode(row['input_ids']), f"{len(row['input_ids'])} tokens")
        print('\n--\n')
    return {'user_id' : user_id}


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


def reset_splits(all_ds, p_train=0.9, p_dev=0.05):

    all_users = list(set(all_ds['user_id']))
    non_train_users = sample(all_users, int(len(all_users) * (1-p_train)))
    p_dev = p_dev / (1-p_train)
    n = int(len(non_train_users) * p_dev)
    dev_users = non_train_users[:n]
    test_users = non_train_users[n:]

    new_ds = DatasetDict()
    new_ds['train'] = all_ds.filter(lambda row: row['user_id'] not in non_train_users)
    new_ds['dev'] = all_ds.filter(lambda row: row['user_id'] in dev_users)
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


def describe_lens(ds, user=False, minmax=False, split='train'):
    # print(tokenized_user_blog_corpus) 
    if isinstance(ds, DatasetDict): # DatasetDict
        df = ds[split].to_pandas()
    else: # Dataset
        df = ds.to_pandas()

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

blogs_path = "/cronus_data/ssmith/data/blog_corpus"
ds4ud_csv_path = "/cronus_data/ssmith/data/raw_ds4ud_FB.csv"
ds4ud_path = "/cronus_data/ssmith/data/ds4ud_corpus"
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
chunked_dsep_path = base_path + f"chunked_dsep_{str(block_size)}"
chunked_docss_path = base_path + f"chunked_docss_{str(block_size)}"
sample_chunked_docss_path = base_path + f"sample_chunked_docss_{str(block_size)}"
sample_chunked_dsep_path = base_path + f"sample_chunked_dsep_{str(block_size)}"
unchunked_path = base_path + f"unchunked_{str(block_size)}"
unchunked_512_path = base_path + f"unchunked_512"
unchunked_512_sample_path = base_path + f"unchunked_512_sample"


if False: # remove rows of 5 tokens or less
    ds = load_from_disk(unchunked_512_sample_path)
    describe_lens(ds)
    new_ds = remove_short_docs(ds, 5)
    describe_lens(new_ds)
    new_ds.save_to_disk(unchunked_512_sample_path + "_NEW")


if False: # sample from unchunked with same users as the dsep sample
    ds = load_from_disk(unchunked_512_path)
    ref_ds = load_from_disk(sample_chunked_dsep_path)
    sample_ds_train = sample_users_from_other_ds(ds['train'], ref_ds['train'])
    describe_lens(ref_ds); describe_lens(sample_ds_train)
    sample_ds = DatasetDict()
    sample_ds['train'] = sample_ds_train
    for key in ds.keys():
        if key != 'train':
            sample_ds[key] = ds[key]
    print(sample_ds)
    sample_ds.save_to_disk(unchunked_512_sample_path)
    exit()


if False: # check out attn mask
    unchunked_ds = load_from_disk(unchunked_512_path)
    row = unchunked_ds['train'].to_pandas().iloc[0]
    print(type(row['attention_mask']))
    exit()


if False: # find long posts in unchunked
    ds = load_from_disk(unchunked_path)
    df = ds['train'].to_pandas()
    df = df[df['input_ids'].str.len() >= 4000]
    print(df.iloc[0]['user_id']); exit()


if False: # send unchunked posts onto multiple rows if they're over 512
    ds = load_from_disk(unchunked_path)
    user_id = decode_user(ds, from_index=False, user_id=1171643, split='train', n=5)['user_id']
    new_ds = chunk_data(tokenizer, ds, 512, one_post=True)
    decode_user(new_ds, from_index=False, user_id=1171643, split='train', n=15)
    describe_lens(ds)
    describe_lens(new_ds)
    new_ds.save_to_disk(unchunked_512_path)
    exit()



if False: # unchunk full chunked
    ds = load_from_disk(chunked_docss_path)
    dsep_ds = load_from_disk(chunked_dsep_path)
    decode_user(dsep_ds, user_index=803)
    unchunked_ds = unchunk_data(ds, tokenizer)
    decode_user(unchunked_ds, user_index=803)
    describe_lens(dsep_ds)
    describe_lens(unchunked_ds)
    unchunked_ds.save_to_disk(unchunked_path)
    exit()


if False: # docss chunk from docss concat
    concat_ds = load_from_disk(concat_path)
    describe_lens(concat_ds)
    chunked_ds = chunk_data(tokenizer, concat_ds, block_size, prune=True)
    chunked_ds = remove_short_docs(chunked_ds, 500)
    describe_lens(chunked_ds)
    chunked_ds.save_to_disk(chunked_docss_path)



if False: # dsep chunk from dsep concat
    concat_ds = load_from_disk(concat_dsep_path)
    describe_lens(concat_ds)
    chunked_ds = chunk_data(tokenizer, concat_ds, block_size, prune=True)
    chunked_ds = remove_short_docs(chunked_ds, 500)
    describe_lens(chunked_ds)
    chunked_ds.save_to_disk(chunked_dsep_path)


if False: # concat to dsep
    ds = load_from_disk(concat_path)
    describe_lens(ds)
    necls_ds = remove_extra_cls_data(ds, tokenizer)
    describe_lens(necls_ds)
    dsep_ds = add_sep(necls_ds, tokenizer)
    describe_lens(dsep_ds)
    dsep_ds.save_to_disk(concat_dsep_path)
    exit()


if False: # send unchunked posts onto multiple rows if they're over 512
    ds = load_from_disk(unchunked_path)
    describe_lens(ds)
    # user_id = decode_user(ds, from_index=True, user_index=778, split='train', n=2)['user_id']
    new_ds = chunk_data(tokenizer, ds, 512, one_post=True)
    describe_lens(new_ds)
    # decode_user(new_ds, from_index=False, user_id=user_id, split='train', n=10)


if False: # unchunk a sample
    ds = load_from_disk(sample_chunked_docss_path)
    describe_lens(ds)
    # sample_decode(ds, index_list=[100, 200, 300])
    unchunked_ds = unchunk_data(ds, tokenizer)
    sample_decode(unchunked_ds, index_list=[1000, 1001, 1002])
    describe_lens(unchunked_ds)
    unchunked_ds.save_to_disk(unchunked_path)
    # unchunked_ds = chunk_data(tokenizer, unchunked_ds, 512, multiple_rows=True)
    # describe_lens(unchunked_ds)
    # sample_decode(unchunked_ds, index_list=[1000, 1001, 1002])

if False: # rename validation to dev
    for path in [sample_concat_path, sample_concat_dsep_path, sample_chunked_path, sample_chunked_dsep_path]:
        ds = load_from_disk(path)
        print(ds)
        new_ds = DatasetDict()
        new_ds['train'] = ds['train']
        new_ds['dev'] = ds['validation']
        new_ds['test'] = ds['test']
        print(new_ds)
        new_ds.save_to_disk(path + "_NEW")


if False: # chunk and remove short from concat
    concat_ds = load_from_disk(concat_path)
    chunked_ds = chunk_data(tokenizer, concat_ds, block_size, prune=True)
    chunked_ds = remove_short_docs(chunked_ds, 500)
    describe_lens(chunked_ds)


if False: # lens of blogs and UD
    ds = load_from_disk(split_path)
    blogs_ds, ud_ds = separate_blogs_ud(ds)
    describe_lens(blogs_ds)
    describe_lens(ud_ds)


if False: # load ud and blogs and check user ids
    ud_ds = load_from_disk(ds4ud_path)
    blogs_ds = load_from_disk(blogs_path)
    print(pd.Series(ud_ds['train'].to_pandas()['user_id'].apply(int).unique()).describe())
    print(pd.Series(blogs_ds['train'].to_pandas()['user_id'].apply(int).unique()).describe())
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
    nv_dev_df = remove_verbose_users(ds['dev'])
    nv_test_df = remove_verbose_users(ds['test'])
    nv_ds = df_to_ds(nv_train_df, nv_dev_df, nv_test_df)
    print("\nafter nv:")
    describe_lens(nv_ds, user=True)
    concat_train_df = concat_df(nv_ds['train']) 
    concat_dev_df = concat_df(nv_ds['dev'])
    concat_test_df = concat_df(nv_ds['test'])
    concat_ds = df_to_ds(concat_train_df, concat_dev_df, concat_test_df)
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
    concat_dev_df = concat_df(nv_ds['dev'])
    concat_test_df = concat_df(nv_ds['test'])
    concat_nv_ds = df_to_ds(concat_train_df, concat_dev_df, concat_test_df)
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
    nv_dev_df = remove_verbose_users(ds['validation'])
    nv_test_df = remove_verbose_users(ds['test'])
    nv_ds = df_to_ds(nv_train_df, nv_dev_df, nv_test_df)
    describe_lens(nv_ds)
    nv_ds.save_to_disk(nv_path)
    

if False:
    ds = load_from_disk(tokenized_path)
    sorted_all_ds = fixed_date_all_df(ds)
    sorted_all_ds.save_to_disk(sorted_all_path)

if False:
    ds = load_from_disk(tokenized_path)
    train_df = ds['train'].to_pandas()    
    dev_df = ds['validation'].to_pandas()
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
