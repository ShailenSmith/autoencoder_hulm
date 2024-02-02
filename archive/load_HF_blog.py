import transformers
import torch
from datasets import load_dataset

blog_corpus = load_dataset('blog_authorship_corpus')
blog_corpus.save_to_disk("/chronos_data/ssmith/blog_corpus")
