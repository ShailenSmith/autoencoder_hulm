# Autoencoder Human Language Modeling (HuLM)
A senior Honors Thesis project with the Human Language Analysis Lab (HLAB) at Stony Brook University. A collection of scripts used to process data and run modeling experiments for a human-aware autoencoder.

## Files
`load_raw_blogs.py` - Converts the blog corpus from a .zip of xml files to a HuggingFace dataset. \
`add_ds4ud.py` - Loads in a csv of the DS4UD Facebook data and concatenates it to the blogs HF set. \
`process.py` - Collection of text processing methods for tokenization, concatenation of user text, filtering by document length, etc. \
`trials.py` - Training loop with Optuna hyperparameter tuning for comparison experiments.

### Process
```$DATA``` here refers to the location of the data folder on the Cronus server (for HLAB members).

`raw_to_HF_blogs.py`: takes ```$DATA/raw_blogs``` and turns into HuggingFace DatasetDict object (located at $DATA/blog_corpus/). \
`add_ds4ud.py`: takes ```$DATA/raw_ds4ud_FB.csv```, saves ds4ud HF DatasetDict object to ```$DATA/ds4ud_corpus/```, and saves the blogs and ds4ud data together in $DATA/blogsUD/blogsUD_corpus/. \
`process.py`: main data preprocessing script that can handle tokenization, concatenation of text by user, filtering by text length, etc. The methods are meant to be called one by one in the MAIN section in the bottom of the script based on what you need to do, so that you can switch steps around or change things without much trouble.

### Run Training
`run_hulm_train.py`: Training loop used to train the full human-aware extended-context Distil-RoBERTa model. \
`run_traditional_train.py`: Training loop used to train the full traditional Distil-RoBERTa model.

### Trials for Hyperparameter Tuning
Some of these files use ```trainer.hyperparameter_search()```, and some do not. The implementation of the ones that do use it should be cleaner.
`docss_trials.py`: Hyperparameter tuning for the DOCSS special token approach. \
`dsep_trials.py`: Hyperparameter tuning for the DSEP special token approach. \
`pos_embd_trials.py`: Hyperparameter tuning for the three positional embeddings initialization techniques. \
`traditional_trials.py`: Hyperparameter tuning for the un-modified Distil-RoBERTa model.

### Modified ```transformers``` Library Files
All areas in these files with changes are marked with \#\#\# in the files.

**modeling_roberta.py**: The file that contains RoBERTa's architecture. There is an added method to the model class that implements the three positional embeddings initialization approaches tested. \
`trainer.py`: The ```Trainer``` used for all the training loops and hyperparameter trials. There are a few modifications, some relying on environment variables, that allow for mask extracton, insertion of custom masks, and extraction of post-softmax probabilities.

### Evaluate
`evaluate.py`: Main prediction and evaluation loop for our Distil-RoBERTa models using the Hugging Face ```trainer```. Combined with the modified trainer.py and data_collator.py in this repo, it can handle prediction, evaluation, extraction of masks, insertion of new masks, and extraction of post-softmax target probabilities to avoid the OOM that often comes with ```trainer.predict()```. \
`compute_ppl.py`: Computes perplexity at the document-level and user-level given a dataset of post-softmax target probabilities.

Please email me at `shailenkent27@gmail.com` with any questions about the code or the project.
