import os
import numpy as np

from Tokenizer import Tokenizer
from Transformer import Transformer
from Layer_Normalization import Layer_Normalization
from config import SAVE_PATH, TRAIN_PATH
from Vocabularizer import Tokenize

DIM_INPUT_VECTOR = 256
NUM_HEAD = 4
DIM_HEAD = 64
NUM_TOKENS = 16
DIM_QUERY = 32

class GPT:
    def __init__(self):
        self.tokenizer = Tokenizer(num_tokens=NUM_TOKENS, dim_input_vector=DIM_INPUT_VECTOR, dim_output_vector=DIM_INPUT_VECTOR)

        self.last_layer_normalizer = Layer_Normalization(num_tokens=NUM_TOKENS)

        self.layers = [
            Transformer(num_tokens=NUM_TOKENS, num_head=NUM_HEAD, dim_head=DIM_HEAD, dim_input_vector=DIM_INPUT_VECTOR, dim_query=DIM_QUERY, feedforward_dims=[DIM_INPUT_VECTOR, 4 * DIM_INPUT_VECTOR, DIM_INPUT_VECTOR]),
            Transformer(num_tokens=NUM_TOKENS, num_head=NUM_HEAD, dim_head=DIM_HEAD, dim_input_vector=DIM_INPUT_VECTOR, dim_query=DIM_QUERY, feedforward_dims=[DIM_INPUT_VECTOR, 4 * DIM_INPUT_VECTOR, DIM_INPUT_VECTOR]),
            Transformer(num_tokens=NUM_TOKENS, num_head=NUM_HEAD, dim_head=DIM_HEAD, dim_input_vector=DIM_INPUT_VECTOR, dim_query=DIM_QUERY, feedforward_dims=[DIM_INPUT_VECTOR, 4 * DIM_INPUT_VECTOR, DIM_INPUT_VECTOR]),
            Transformer(num_tokens=NUM_TOKENS, num_head=NUM_HEAD, dim_head=DIM_HEAD, dim_input_vector=DIM_INPUT_VECTOR, dim_query=DIM_QUERY, feedforward_dims=[DIM_INPUT_VECTOR, 4 * DIM_INPUT_VECTOR, DIM_INPUT_VECTOR]),
            Transformer(num_tokens=NUM_TOKENS, num_head=NUM_HEAD, dim_head=DIM_HEAD, dim_input_vector=DIM_INPUT_VECTOR, dim_query=DIM_QUERY, feedforward_dims=[DIM_INPUT_VECTOR, 4 * DIM_INPUT_VECTOR, DIM_INPUT_VECTOR]),
            Transformer(num_tokens=NUM_TOKENS, num_head=NUM_HEAD, dim_head=DIM_HEAD, dim_input_vector=DIM_INPUT_VECTOR, dim_query=DIM_QUERY, feedforward_dims=[DIM_INPUT_VECTOR, 4 * DIM_INPUT_VECTOR, DIM_INPUT_VECTOR]),
        ]

    def forward_train(self, text):
        input_vectors = self.tokenizer.embed_train(text)

        transformed = input_vectors.copy()
        for i in range(len(self.layers)):
            transformed = self.layers[i].forward_train(transformed)
        
        transformed = self.last_layer_normalizer.forward_train(transformed)

        chosen_token = self.tokenizer.unembed_train(transformed)

        return chosen_token

    def backward(self, expected_last_word):
        dC_dX = self.tokenizer.backward_unembed(expected_last_word)

        current_dC_dX = self.last_layer_normalizer.backward(dC_dX)

        for i in reversed(range(len(self.layers))):
            current_dC_dX = self.layers[i].backward(current_dC_dX)

        self.tokenizer.backward_embed(current_dC_dX)

    def save_layers(self):
        try:
            os.mkdir(SAVE_PATH)
            print(f"Folder {SAVE_PATH} created successfully.")
        except:
            print("")

        np.savetxt(SAVE_PATH + "/embedding.out", self.tokenizer.embedding_matrix)
        np.savetxt(SAVE_PATH + "/unembedding.out", self.tokenizer.unembedding_matrix)

        for i in range(len(self.layers)):
            layer = self.layers[i]

            match layer.type:
                case "transformer":
                    for j in range(layer.attention.num_head):
                        np.savetxt(SAVE_PATH + "/query" + str(i) + str(j) + ".out", layer.attention.heads[j][0])
                        np.savetxt(SAVE_PATH + "/key" + str(i) + str(j) + ".out", layer.attention.heads[j][1])
                        np.savetxt(SAVE_PATH + "/value" + str(i) + str(j) + ".out", layer.attention.heads[j][2])
                    np.savetxt(SAVE_PATH + "/output" + str(i) + ".out", layer.attention.output_matrix)
                    for j in range(layer.multilayer_feedforward.num_layers):
                        for k in range(len(layer.multilayer_feedforward.dims) - 1):
                            np.savetxt(SAVE_PATH + "/weights" + str(i) + str(j) + str(k) + ".out", layer.multilayer_feedforward.layers[j].weights[k])
                            np.savetxt(SAVE_PATH + "/biases" + str(i) + str(j) + str(k) + ".out", layer.multilayer_feedforward.layers[j].biases[k])
                    np.savetxt(SAVE_PATH + "/gamma" + str(i) + "1.out", layer.layer_normalizer1.GAMMA)
                    np.savetxt(SAVE_PATH + "/beta" + str(i) + "1.out", layer.layer_normalizer1.BETA)
                    np.savetxt(SAVE_PATH + "/gamma" + str(i) + "2.out", layer.layer_normalizer2.GAMMA)
                    np.savetxt(SAVE_PATH + "/beta" + str(i) + "2.out", layer.layer_normalizer2.BETA)

    def load_layers_if_exist(self):
        if not os.path.isdir(SAVE_PATH + "/embedding.out"):
            return

        self.tokenizer.embedding_matrix = np.loadtxt(SAVE_PATH + "/embedding.out")
        self.tokenizer.unembedding_matrix = np.loadtxt(SAVE_PATH + "/unembedding.out")

        for i in range(len(self.layers)):

            match self.layers[i].type:
                case "transformer":
                    for j in range(self.layers[i].attention.num_head):
                        self.layers[i].attention.heads[j][0] = np.loadtxt(SAVE_PATH + "/query" + str(i) + str(j) + ".out")
                        self.layers[i].attention.heads[j][1] = np.loadtxt(SAVE_PATH + "/key" + str(i) + str(j) + ".out")
                        self.layers[i].attention.heads[j][2] = np.loadtxt(SAVE_PATH + "/value" + str(i) + str(j) + ".out")
                    self.layers[i].attention.output_matrix = np.loadtxt(SAVE_PATH + "/output" + str(i) + ".out")
                    for j in range(self.layers[i].multilayer_feedforward.num_layers):
                        for k in range(len(self.layers[i].multilayer_feedforward.dims) - 1):
                            self.layers[i].multilayer_feedforward.layers[j].weights[k] = np.loadtxt(SAVE_PATH + "/weights" + str(i) + str(j) + str(k) + ".out")
                            self.layers[i].multilayer_feedforward.layers[j].biases[k] = np.loadtxt(SAVE_PATH + "/biases" + str(i) + str(j) + str(k) + ".out")
                    self.layers[i].layer_normalizer1.GAMMA = np.loadtxt(SAVE_PATH + "/gamma" + str(i) + "1.out")
                    self.layers[i].layer_normalizer1.BETA = np.loadtxt(SAVE_PATH + "/beta" + str(i) + "1.out")
                    self.layers[i].layer_normalizer2.GAMMA = np.loadtxt(SAVE_PATH + "/gamma" + str(i) + "2.out")
                    self.layers[i].layer_normalizer2.BETA = np.loadtxt(SAVE_PATH + "/beta" + str(i) + "2.out")

    def train_on_text(self):
        text = open(TRAIN_PATH).read()
        tokens = Tokenize(text, self.tokenizer.vocabulary, num_tokens=None)

        for i in range(1, len(tokens)):
            self.forward_train(" ".join(tokens[0 : i]))
            self.backward(tokens[i])

            if i % 100 == 0:
                print("TEST RUN: " + self.generate_from_text("He", 100))

    def generate_from_text(self, input_text, iterations):
        output_text = input_text
        for i in range(iterations):
            output_text = output_text + " " + self.forward_train(output_text)
        return output_text

gpt_network = GPT()
gpt_network.load_layers_if_exist()

EPOCHS = 5
for i in range(EPOCHS):
    gpt_network.train_on_text()
    gpt_network.save_layers()
