import numpy as np
import random
import math

from Regularizer import Smooth_Label, Normalize_Factor, Dropout, KEEP_PROB
from Activation import Softmax, Softmax_derivative
from Adam import Adam

class Tokenizer:
    def __init__(self, num_tokens, dim_input_vector, dim_output_vector):
        self.type = "tokenizer"

        self.num_tokens = num_tokens
        self.dim_input_vector = dim_input_vector
        self.dim_output_vector = dim_output_vector

        self.embed_scaling = math.sqrt(dim_input_vector)

        text = open("./vocabulary-text.txt").read()
        self.vocabulary = list(set(text.split()))
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
        words = text.split()

        input_vectors = np.empty((self.num_tokens, self.dim_input_vector))

        if len(words) > self.num_tokens - 1:
            words = words[len(words) - self.num_tokens + 1 : len(words)]
        elif len(words) < self.num_tokens - 1:
            words = ["<empty>"] * (self.num_tokens - 1 - len(words)) + words

        self.word_indices = []
        for i in range(self.num_tokens - 1):
            index = self.index_of_token(words[i])
            self.word_indices.append(index)
            input_vectors[i, :] = self.embedding_matrix[index, :]

        empty_vector = self.embedding_matrix[-1, :]
        input_vectors[-1, :] = empty_vector
        # the vector to predict

        input_vectors *= self.embed_scaling

        input_vectors += self.PE

        input_vectors = Dropout(input_vectors)

        return input_vectors

    def unembed_train(self, output_vectors):
        self.output_vectors = output_vectors

        last_vector = output_vectors[-1, :]

        preactivations = np.matmul(last_vector, self.unembedding_matrix)
        preactivations = preactivations - np.max(preactivations)

        # softmax
        prob_dist = Softmax(preactivations)
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

    def backward_embed(self, dC_dY, LEARNING_RATE):
        # Update word embedding vector
        for i in range(len(self.word_indices)):
            index = self.word_indices[i]

            dC_dEmbed = dC_dY[i, :]
            embed_adam = Adam(dC_dEmbed, self.embed_m_prev[index, :], self.embed_v_prev[index, :], LEARNING_RATE)
            self.embedding_matrix[index, :] += embed_adam[0] * self.embed_scaling / KEEP_PROB
            self.embed_m_prev[index, :] = embed_adam[1]
            self.embed_v_prev[index, :] = embed_adam[2]

    def backward_unembed(self, expected_last_word, LEARNING_RATE, loss_function_back):
        expected_prob_dist = Smooth_Label(len(self.vocabulary), self.index_of_token(expected_last_word), 0.0001)

        # Derivative of cross entropy between expected prob dist and predicted prob dist
        dC_dP = loss_function_back(self.prob_dist, expected_prob_dist)

        dP_dpre = Softmax_derivative(self.prob_dist)

        # Derivative cost with pre-softmax, by composing dC_dP and dP_dpre
        dC_dpre = np.zeros((self.num_tokens, len(self.vocabulary)))
        dC_dpre[-1, :] = np.sum(dP_dpre * dC_dP, axis=1)

        dC_dUnembedding = np.matmul(np.transpose(self.output_vectors), dC_dpre)
        dC_dX = np.matmul(dC_dpre, np.transpose(self.unembedding_matrix))

        unembed_adam = Adam(dC_dUnembedding, self.unembed_m_prev, self.unembed_v_prev, LEARNING_RATE)
        self.unembedding_matrix += unembed_adam[0]
        self.unembed_m_prev = unembed_adam[1]
        self.unembed_v_prev = unembed_adam[2]

        dC_dX *= Normalize_Factor(dC_dX)

        return dC_dX
