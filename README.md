# MING-GPT

Minimal GPT based on the transformer model, written in pure Numpy. Still under development.

## Architecture

### Components
Tokenizer: Word embedding and positional encoding (sine).
Transformer: Layer normalization, attention (masking), layer normalization, feedforward, layer normalization (last transformer only).
Tokenizer: Word unembedding.
Cost function: Cross entropy.

### Structure
Tokenizer embedding -> 8 Transformers -> Tokenizer unembedding.

### Regularization and Optimization
Adam optimizer.
Gradient normalization for every attention, feedforward and layer normalization.
Layer normalization (pre).
Label smoothing.
Dropout after attention and feedforward layers.

## Usage

The main file is `GPT.py`. A network is already set up inside, ready to be customized. The vocabulary is extracted from `vocabulary-text.txt`, which must contain the `<empty>` and `<unk>` tokens at the end in this order. The model trains using the `train-text.txt` file. It reads it and tries to predict the next word, for each word in the text.

## Notes

Currently training the model on random Ancient Rome story.
