"""Evaluate and predict a trained distil-RoBERTa model (WIP)"""

from transformers import pipeline
from transformers import RobertaForMaskedLM
from transformers import AutoTokenizer
import sys
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
import math

grouped_blog_corpus = load_from_disk("/chronos_data/ssmith/data/grouped_blog_corpus_128")

model_path = "/chronos_data/ssmith/models/blogBERTa_8_256"

model = RobertaForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

fill_mask = pipeline(
            "fill-mask",
            model=model,
            tokenizer="roberta-base"
            )

if len(sys.argv) == 2:
    result = fill_mask(sys.argv[1])
    print([f"{row['token_str']}, {row['score']:.3f}" for row in result])

training_args = TrainingArguments(
        output_dir=f"{model_path}/eval",
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=grouped_blog_corpus['validation'],
        data_collator=data_collator,
)

# result = trainer.predict(test_dataset=grouped_blog_corpus['validation'])
# print(result['predictions'])

# trainer.save_model("/chronos_data/ssmith/data/test_pred_save")


print("running trainer.evaluate():")
eval_results = trainer.evaluate()

print(eval_results)
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

