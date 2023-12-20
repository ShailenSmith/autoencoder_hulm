"""WIP for training a distil-RoBERTa model from a checkpoint."""

import transformers
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers.models.roberta.configuration_roberta import *
from transformers.models.roberta.modeling_roberta import *
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
import torch
import pandas as pd
import os


def train(batch_size, block_size, version, small=False):

    # load saved dataset
    print("loading dataset...")
    grouped_blog_corpus = load_from_disk(f"/chronos_data/ssmith/data/user_blogs/grouped_blog_corpus_{block_size}")
    print("dataset loaded")

    for i in range(10):
        print(len(grouped_blog_corpus['train'][i]['input_ids']))

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    small_config_dict = {
            'vocab_size' : 50265, # number of total tokens allowed
            'num_hidden_layers' : 2, # number of hidden RobertaLayers in a RobertaEncoder
            'num_attention_heads' : 4, # multi-headed attention heads
            'hidden_size' : 64,
            'intermediate_size' : 256, # dimension of feedfoward layer in encoder
            'hidden_act' : "gelu", # nonlinearity in the encoder and pooler
            'hidden_dropout_prob' : 0.1, # dropout probability for fully conn. layers
            'attention_probs_dropout_prob' : 0.1, # dropout in attn layer
            'max_position_embeddings' : block_size+2, # max seq. length the model could ever have
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

    # load config and model
    normal_config_dict = {
            'vocab_size' : 50265, # number of total tokens allowed
            'num_hidden_layers' : 6, # number of hidden RobertaLayers in a RobertaEncoder
            'num_attention_heads' : 12, # multi-headed attention heads
            'hidden_size' : 768, # dimension of hidden layers
            'intermediate_size' : 3072, # dimension of feedfoward layer in encoder
            'max_position_embeddings' : block_size+2, # max seq. length the model could ever have
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

    base_output_dir = f"/cronus_data/ssmith/models/user_models/blogBERTa_{batch_size}_{block_size}"
    
    config_dict = small_config_dict if small else normal_config_dict
    roberta_config = RobertaConfig(**config_dict)
    model = RobertaForMaskedLM.from_pretrained(base_output_dir)

    # distil = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
    # print(tokenizer.decode([0, 1, 2, 3]))
    # print(distil, distil.config); exit()


    # training arguments
    training_args = TrainingArguments(
        output_dir = base_output_dir,
        overwrite_output_dir=False,
        logging_strategy="steps",
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=5000,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        save_steps=5000,
        save_total_limit=2,
        prediction_loss_only=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
    )


    # data collator - performs batching and masking (i think)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


    # print("------- MODEL ----------", model)
    print(f"number of parameters: {model.num_parameters()}")
    print('------------------------')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped_blog_corpus['train'],
        eval_dataset=grouped_blog_corpus['validation'],
        data_collator = data_collator,
    )

    trainer.train()
    trainer.save_model()

# ------------------- Main method ---------------------------

gpus = [2]
block_sizes = [4096]
version = [2]
small = [False]

for i in range(len(gpus)):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i])
    train(batch_size=1, block_size=block_sizes[i], version=version[i], small=small[i])
