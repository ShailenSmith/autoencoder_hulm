# Autoencoder Human Language Modeling (HuLM)
Senior Honors Thesis project with the Human Language Analysis Lab (HLAB). A collection of scripts used to process data and run modeling experiments for a human-aware autoencoder. \
\
**Data used:** Blog Authorship Corpus and DS4UD Facebook data (TODO add more info). \
\
**Methodology:** TODO add

## Files
**load_raw_blogs.py** - Converts the blog corpus from a .zip of xml files to a HuggingFace dataset. \
**add_ds4ud.py** - Loads in a csv of the DS4UD Facebook data and concatenates it to the blogs HF set. \
**process.py** - Collection of text processing methods for tokenization, concatenation of user text, filtering by document length, etc. \
**blog_roberta_trainer.py** - Main training loop for fine-tuning a distil-RoBERTa model with an expanded context size. \
**trials.py** - Training loop with Optuna hyperparameter tuning for comparison experiments. \
**evaluate_roberta.py** - Prediction and evaluation script for trained distil-RobERTa models (WIP). \
**check_weights.py** - Script for analyzing a trained distil-RoBERTa model's learned weights (WIP). \
**plot_tb.py** - Converting tensorboard file data into a csv (WIP).
