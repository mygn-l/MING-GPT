# MING-GPT

Minimal GPT written in Numpy.

## Usage

Before anything, generate a vocabulary with `Gen_Vocab.py`.
The main file `GPT.py` contains a pre-setup model and can be run immediately after generating vocabulary. Some configs are in the `__init__` block of GPT class.
All other configurations can be seen and changed in `config.py`.

Regenerating a vocabulary will ruin your training progress, so resign yourself to a good vocabulary from the get-go.
You can start anew (throw away all progress and restart) by deleting the appropriate`SAVE_PATH` folder (initialized to `MING-GPT`). You can also choose to keep it, and instead, change the `SAVE_PATH` to something else; the model will restart anew in that new folder, while the old folder retains progress, and can be switched back at any time.
