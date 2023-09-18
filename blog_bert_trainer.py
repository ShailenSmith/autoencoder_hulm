import transformers
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

# load saved dataset
batched_blog_corpus = load_from_disk("/data/batched_blog_corpus")

# model and tokenizer name
model_checkpoint = "bert-base-uncased"
tokenizer_checkpoint = "bert-base-uncased"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# load config and model
config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_config(config)

# training arguments
training_args = TrainingArguments(
    "test-mlm",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

# data collator - performs batching and masking (i think)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

print(model)
print('--', '--')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=batched_blog_corpus['train'],
    eval_dataset=batched_blog_corpus['validation'],
    data_collator = data_collator,
)

def main():
    trainer.train()

if __name__ == "__main__":
    main()