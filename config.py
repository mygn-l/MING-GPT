import math

# Temperature
TEMPERATURE = 1

# The file from which to generate vocabulary
VOCAB_PATH = "./train-text.txt"

# The training file
TRAIN_PATH = "./train-text.txt"

# Vocabulary size
VOCAB_SIZE = 500

# Don't touch
CODE_LENGTH = int(math.floor(math.log(VOCAB_SIZE, 10))) + 1
CODE_LENGTH2 = 2 * CODE_LENGTH
ZERO = "0" * CODE_LENGTH
EMPTY_TOKEN = "<empty>"
UNK_TOKEN = "<unk>"

# Model load and save path
SAVE_PATH = "MING-GPT"

# Adam optimizer
LEARNING_RATE = 0.001 #normally between 0.001 and 0.0001
EPSILON = 0.000000001 #do not change, used to avoid division by zero
BETA1 = 0.9 #velocity preservation ratio
BETA2 = 0.98 #acceleration preservation ratio

# Dropout rate
KEEP_PROB = 0.9 #normally 0.9

# Label smoothing
SMOOTHING_FACTOR = 0.001
