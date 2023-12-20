"""WIP for checking model weights."""

from transformers import RobertaForMaskedLM
from datasets import load_from_disk
from transformers import AutoTokenizer
import torch

block_size = 4096
batch_size = 1

grouped_blog_corpus = load_from_disk(f"/chronos_data/ssmith/data/user_blogs/grouped_blog_corpus_{block_size}")

model_path = f"/chronos_data/ssmith/models/user_models/blogBERTa_{batch_size}_{block_size}"
# small_model_path = "/chronos_data/ssmith/models/user_models/blogBERTa-small"
# large_model_path = "/chronos_data/ssmith/models/blogBERTa-large"

model = RobertaForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# print(model)
# exit()

for i in range(model.config.num_hidden_layers):
    queries = model.roberta.encoder.layer[i].attention.self.query.weight
    int_fully_conn = model.roberta.encoder.layer[i].intermediate.dense.weight
    # print(int_fully_conn[0][:10])

def check_word_embeddings(model, tokens):
    tokens = torch.LongTensor(tokens)
    word_embeddings = model.roberta.embeddings.word_embeddings
    print(word_embeddings(tokens).shape)

tokens = tokenizer.encode("The dog and the cat and the dog and the cat.")
check_word_embeddings(model, tokens)
# print(tokens)
# print(tokenizer.decode(tokens))
