import json
import os
import time

import numpy as np

from Tokenizer import Tokenizer
from Transformer import Transformer

DIRECTORY = "MING-GPT"
LEARNING_RATE = 0.1

class GPT:
    def __init__(self):
        self.tokenizer = Tokenizer(num_tokens=8, dim_input_vector=128, dim_output_vector=128)

        self.transformer1 = Transformer(num_tokens=8, num_head=8, dim_head=16, dim_input_vector=128, dim_query=8, feedforward_dims=[128, 512, 128])
        self.transformer2 = Transformer(num_tokens=8, num_head=8, dim_head=16, dim_input_vector=128, dim_query=8, feedforward_dims=[128, 512, 128])
        self.transformer3 = Transformer(num_tokens=8, num_head=8, dim_head=16, dim_input_vector=128, dim_query=8, feedforward_dims=[128, 512, 128])
        self.transformer4 = Transformer(num_tokens=8, num_head=8, dim_head=16, dim_input_vector=128, dim_query=8, feedforward_dims=[128, 512, 128])
        self.transformer5 = Transformer(num_tokens=8, num_head=8, dim_head=16, dim_input_vector=128, dim_query=8, feedforward_dims=[128, 512, 128])
        self.transformer6 = Transformer(num_tokens=8, num_head=8, dim_head=16, dim_input_vector=128, dim_query=8, feedforward_dims=[128, 512, 128])

        self.layers = [self.transformer1, self.transformer2, self.transformer3, self.transformer4, self.transformer5, self.transformer6]

    def forward_train(self, text):
        input_vectors = self.tokenizer.embed_train(text)

        transformed = input_vectors.copy()
        for i in range(len(self.layers)):
            transformed = self.layers[i].forward_train(transformed)

        chosen_token = self.tokenizer.unembed_train(transformed)

        return chosen_token

    def backward(self, expected_last_word):
        dC_dX = self.tokenizer.backward_unembed_crossentropy(expected_last_word, LEARNING_RATE)

        current_dC_dX = dC_dX.copy()
        for i in reversed(range(len(self.layers))):
            current_dC_dX = self.layers[i].backward(current_dC_dX, LEARNING_RATE)

        self.tokenizer.backward_embed(current_dC_dX, LEARNING_RATE)

    def save_layers(self):
        try:
            os.mkdir(DIRECTORY)
            print("Directory created successfully.")
        except:
            print("")

        for i in range(len(self.layers)):
            layer = self.layers[i]

            match layer.type:
                case "tokenizer":
                    np.savetxt(DIRECTORY + "/embedding.out", layer.embedding_matrix)
                    np.savetxt(DIRECTORY + "/unembedding.out", layer.unembedding_matrix)
                    with open(DIRECTORY + "/vocabulary.json", "w") as file:
                        file.write(json.dumps(layer.vocabulary))

                case "transformer":
                    for j in range(layer.attention.num_head):
                        np.savetxt(DIRECTORY + "/query" + str(i) + str(j) + ".out", layer.attention.heads[j][0])
                        np.savetxt(DIRECTORY + "/key" + str(i) + str(j) + ".out", layer.attention.heads[j][1])
                        np.savetxt(DIRECTORY + "/value" + str(i) + str(j) + ".out", layer.attention.heads[j][2])
                    for j in range(layer.multilayer_feedforward.num_layers):
                        for k in range(len(layer.multilayer_feedforward.dims) - 1):
                            np.savetxt(DIRECTORY + "/weights" + str(i) + str(j) + str(k) + ".out", layer.multilayer_feedforward.layers[j].weights[k])
                            np.savetxt(DIRECTORY + "/biases" + str(i) + str(j) + str(k) + ".out", layer.multilayer_feedforward.layers[j].biases[k])

    def load_layers(self):
        DIRECTORY = "MING-GPT"

        for i in range(len(self.layers)):
            layer = self.layers[i]

            match layer.type:
                case "tokenizer":
                    layer.embedding_matrix = np.loadtxt(DIRECTORY + "/embedding.out")
                    layer.unembedding_matrix = np.loadtxt(DIRECTORY + "/unembedding.out")
                    layer.vocabulary = json.loads(open(DIRECTORY + "/vocabulary.json").read())

                case "transformer":
                    for j in range(layer.attention.num_head):
                        layer.attention.heads[j][0] = np.loadtxt(DIRECTORY + "/query" + str(i) + str(j) + ".out")
                        layer.attention.heads[j][1] = np.loadtxt(DIRECTORY + "/key" + str(i) + str(j) + ".out")
                        layer.attention.heads[j][2] = np.loadtxt(DIRECTORY + "/value" + str(i) + str(j) + ".out")
                    for j in range(layer.multilayer_feedforward.num_layers):
                        for k in range(len(layer.multilayer_feedforward.dims) - 1):
                            layer.multilayer_feedforward.layers[j].weights[k] = np.loadtxt(DIRECTORY + "/weights" + str(i) + str(j) + str(k) + ".out")
                            layer.multilayer_feedforward.layers[j].biases[k] = np.loadtxt(DIRECTORY + "/biases" + str(i) + str(j) + str(k) + ".out")

    def train_on_text(self, path):
        text = open(path).read().lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").replace("’", " ").replace("-", " ").replace(":", " ").replace(";", " ").replace("(", " ").replace(")", " ").replace('"', " ").replace("'", " ")
        words = text.split()

        start = time.time()

        for i in range(1, len(words)):
            self.forward_train(" ".join(words[0 : i]))
            self.backward(words[i])
            time.sleep(0.3)
            print("Trained word")

        end = time.time()

        print(f"Trained {len(words)} words in: {end - start} seconds")

    def generate_from_text(self, input_text, iterations):
        output_text = input_text
        for i in range(iterations):
            output_text = output_text + " " + self.forward_train(output_text)
        return output_text

gpt_network = GPT()
if os.path.isdir("./MING-GPT"):
    gpt_network.load_layers()
#print("Started training")
for i in range(1):
    #gpt_network.train_on_text("./train-text.txt")
    print(gpt_network.generate_from_text("I", 100))
    #gpt_network.save_layers()
