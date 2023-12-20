"""Load raw blogs data as a HF dataset."""

from datasets import load_dataset

blog_corpus = load_dataset('/cronus_data/ssmith/data/raw_user_blogs')
print(blog_corpus)

# blog_corpus.save_to_disk("/chronos_data/ssmith/data/user_blog_corpus")
