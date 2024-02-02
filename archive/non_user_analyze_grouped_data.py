from datasets import load_from_disk
from transformers import AutoTokenizer

block_size = 4096
grouped_blog_corpus = load_from_disk(f"/chronos_data/ssmith/data/user_blogs/grouped_blog_corpus_{block_size}")

# tokenized_blog_corpus = load_from_disk("/cronus_data/ssmith/data/tokenized_blog_corpus")

# blog_corpus = load_from_disk("/chronos_data/ssmith/data/blog_corpus")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# train = blog_corpus['train']
grouped_train = grouped_blog_corpus['train']




print(tokenizer.decode(grouped_train[0]['input_ids']), '\n')
exit()

print("\n -- \n")
for i in [0, 1, 2, 3]:
    print(tokenizer.decode(tokenized_train[i]['input_ids']), '\n')
