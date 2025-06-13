import numpy as np
import random
import math
import json

from Regularizer import Smooth_Label, Dropout
from Activation import Softmax
from Adam import Adam
from config import TEMPERATURE, KEEP_PROB, SAVE_PATH
from Vocabularizer import Tokenize

class Tokenizer:
    def __init__(self, num_tokens, dim_input_vector, dim_output_vector):
        self.type = "tokenizer"

        self.num_tokens = num_tokens
        self.dim_input_vector = dim_input_vector
        self.dim_output_vector = dim_output_vector

        self.embed_scaling = math.sqrt(dim_input_vector)

        VOCABULARY = json.loads(open(SAVE_PATH + "/vocabulary.json").read())

        self.vocabulary = VOCABULARY
        self.embedding_matrix = np.random.rand(len(self.vocabulary), dim_input_vector) - 0.5
        self.unembedding_matrix = np.random.rand(dim_output_vector, len(self.vocabulary)) - 0.5

        #positional embedding
        PE = np.array([[pos / (10000 ** (i / dim_input_vector)) for i in range(dim_input_vector)] for pos in range(num_tokens)])
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        self.PE = PE

        self.embed_m_prev = np.zeros((len(self.vocabulary), dim_input_vector))
        self.embed_v_prev = np.zeros((len(self.vocabulary), dim_input_vector))
        self.unembed_m_prev = np.zeros((dim_output_vector, len(self.vocabulary)))
        self.unembed_v_prev = np.zeros((dim_output_vector, len(self.vocabulary)))

    def index_of_token(self, token):
        return self.vocabulary.index(token) if token in self.vocabulary else -1

    def embed_train(self, text):
        tokens = Tokenize(text, self.vocabulary, self.num_tokens)

        embeddings = np.empty((self.num_tokens, self.dim_input_vector))

        self.used_word_indices = []
        # Find word embeddings
        for i in range(self.num_tokens - 1):
            index = self.index_of_token(tokens[i])
            self.used_word_indices.append(index)
            embeddings[i, :] = self.embedding_matrix[index, :]

        # Add <empty> token at end, which is to be predicted
        empty_vector = self.embedding_matrix[-1, :]
        embeddings[-1, :] = empty_vector

        input_vectors = Dropout(embeddings * self.embed_scaling + self.PE)

        return input_vectors

    def backward_embed(self, dC_dY):
        dC_dEmbed = dC_dY * self.embed_scaling / KEEP_PROB
        # Update embedding matrix
        for i in range(self.num_tokens - 1):
            dC_dEmbed_i = dC_dEmbed[i, :]
            index = self.used_word_indices[i]

            embed_adam = Adam(dC_dEmbed_i, self.embed_m_prev[index, :], self.embed_v_prev[index, :])
            self.embedding_matrix[index, :] += embed_adam[0]
            self.embed_m_prev[index, :] = embed_adam[1]
            self.embed_v_prev[index, :] = embed_adam[2]

    def unembed_train(self, output_vectors):
        self.output_vectors = output_vectors

        last_vector = output_vectors[-1, :]

        preactivations = last_vector @ self.unembedding_matrix
        preactivations /= TEMPERATURE

        # softmax
        prob_dist = Softmax(preactivations.reshape((1, -1)))[0, :]
        self.prob_dist = prob_dist

        # choose randomly from the probability distibution
        random_number = random.random()
        current_sum = 0
        index = 0
        while current_sum <= random_number:
            current_sum += self.prob_dist[index]
            index += 1
        index -= 1

        chosen_token = self.vocabulary[index]

        return chosen_token

    def backward_unembed(self, expected_last_word):
        expected_prob_dist = Smooth_Label(len(self.vocabulary), self.index_of_token(expected_last_word))

        """
        # Derivative of cross entropy between expected prob dist and predicted prob dist
        dC_dP = COST_FUNCTION_DERIVATIVE(self.prob_dist, expected_prob_dist)

        dP_dpre = Softmax_derivative(self.prob_dist)

        # Derivative cost with pre-softmax, by composing dC_dP and dP_dpre
        dC_dpre = np.zeros((self.num_tokens, len(self.vocabulary)))
        dC_dpre[-1, :] = dP_dpre @ dC_dP
        """

        dC_dpre = np.zeros((self.num_tokens, len(self.vocabulary)))
        # Using online derivation, mine doesn't work for some reason
        dC_dpre[-1, :] = self.prob_dist - expected_prob_dist

        dC_dUnembedding = self.output_vectors.T @ dC_dpre
        dC_dX = dC_dpre @ self.unembedding_matrix.T

        unembed_adam = Adam(dC_dUnembedding, self.unembed_m_prev, self.unembed_v_prev)
        self.unembedding_matrix += unembed_adam[0]
        self.unembed_m_prev = unembed_adam[1]
        self.unembed_v_prev = unembed_adam[2]

        return dC_dX
