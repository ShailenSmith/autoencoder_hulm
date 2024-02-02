"""Load in the raw DS4UD FB set csv and concatenate with the blogs data."""


from datasets import load_from_disk, concatenate_datasets
from datasets import Dataset, DatasetDict, Value
import csv
import pandas as pd


ds4ud_path = "/cronus_data/ssmith/data/raw_ds4ud_FB.csv"

print("reading csv...")
df = pd.read_csv(ds4ud_path)
print("csv read")

df = df[['user_id', 'message', 'created_time']]
df.dropna(inplace=True) # drop rows where user_id is null
df = df.rename(columns={'message' : 'text', 'created_time' : 'date'})
print(df[['user_id']].describe())


n1 = int(0.95*len(df)) - 100 # start a little before to make time to find new user
print(f'starting n1: {n1}')

# find a train/val split point that doesn't break up a user's messages
for i in range(1000):
    starting_user_id = df.iloc[n1]['user_id']
    if df.iloc[n1 + i]['user_id'] != starting_user_id:
        n1 = n1 + i
        break

print(n1 / len(df))


# train and val split for ds4ud
train_df = df[:n1]
val_df = df[n1:]

print(train_df.columns)
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_ds = Dataset.from_pandas(val_df, preserve_index=False)
ds = DatasetDict()
ds['train'] = train_ds
ds['validation'] = val_ds
if False:
    ds.save_to_disk("/cronus_data/ssmith/data/ds4ud_corpus")

# add blogs
blogs = load_from_disk("/cronus_data/ssmith/data/raw_blogs")
blogs = blogs.remove_columns(['gender', 'age', 'horoscope', 'job'])
blogs_train = blogs['train']
print(blogs_train.to_pandas()['user_id'].describe()); exit()


def make_user_str(row):
    row['user_id'] = str(row['user_id'])

def match_user_types(ds_list):
    new_ds_list = []
    for i in range(len(ds_list)):
        ds = ds_list[i]
        ds.map(make_user_str, num_proc=4)
        new_features = ds.features.copy()
        new_features['user_id'] = Value(dtype='string')
        new_ds_list.append(ds.cast(new_features))
    return new_ds_list

blogs_train, blogs_val, train_ds, val_ds = match_user_types([
    blogs['train'], blogs['validation'], train_ds, val_ds])


# check # unique users in ud set
print(len(set(train_ds['user_id'])))
print(len(set(blogs_train['user_id'])))

assert blogs_train.features.type == train_ds.features.type
combined_train_ds = concatenate_datasets([blogs_train, train_ds])
combined_val_ds = concatenate_datasets([blogs_val, val_ds])

print(len(set(combined_train_ds['user_id'])))


combined_ds = DatasetDict()
combined_ds['train'] = combined_train_ds
combined_ds['validation'] = combined_val_ds

train_users = combined_train_ds['user_id']
val_users = combined_val_ds['user_id']
both_users = set(train_users).intersection(set(val_users))
print(len(both_users)); exit()

combined_ds.save_to_disk("/cronus_data/ssmith/data/blogs_ud/blog_ds4ud_corpus")

