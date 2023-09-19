import transformers
print('transformers ' + transformers.__version__)
from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess(tokenizer_name, dataset, out_path, block_size):

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

  def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True)

  tokenized_data = dataset.map(tokenize_function, batched=True, num_proc=4,
                                      remove_columns=['text', 'date', 'gender',
                                                      'age', 'horoscope', 'job'])
  def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop small remainder
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

  grouped_data = tokenized_data.map(
      group_texts,
      batched=True,
      batch_size=1000,
      num_proc=4,
  )

  grouped_data.save_to_disk(out_path)

def main():
  # print(f"transformers version {transformers.__version__}")

  # tokenizer name
  tokenizer_checkpoint = "bert-base-uncased"

  # load data
  blog_corpus = load_dataset('blog_authorship_corpus')

  # tokenize and batch
  preprocess(tokenizer_checkpoint, blog_corpus, out_path="/data1/ssmith/batched_blog_corpus", block_size=128)


if __name__ == "__main__":
  main()