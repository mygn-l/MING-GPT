# MING-GPT

Minimal GPT based on the transformer model. Still under development.

## Usage

The main file is `GPT.py`. A network is already set up inside to be customized. The vocabulary is extracted from `vocabulary-text.txt`, which must contain the `<empty>` and `<unk>` tokens at the end in this order. The model trains on `train-text` file. It reads it and tries to predict the next word, for each word in the text.

## Notes

I am trying to implement the Adam optimizer and other regularization techniques (such as batch and normalization). Currently, after a few thousand iterations, the model undergoes gradient explosion.
