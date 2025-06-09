import numpy as np
import random

class Tokenizer:
    def __init__(self, num_tokens, dim_input_vector, dim_output_vector):
        self.type = "tokenizer"

        self.num_tokens = num_tokens
        self.dim_input_vector = dim_input_vector
        self.dim_output_vector = dim_output_vector

        text = open("./vocabulary-text.txt").read()
        self.vocabulary = list(set(text.split()))
        self.embedding_matrix = np.random.rand(len(self.vocabulary), dim_input_vector)
        self.unembedding_matrix = np.random.rand(dim_output_vector, len(self.vocabulary))

        #positional embedding
        PE = np.array([[pos / (10000 ** (i / dim_input_vector)) for i in range(dim_input_vector)] for pos in range(num_tokens)])
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        self.PE = PE

    def embed_train(self, text):
        words = text.split()

        input_vectors = np.empty((self.num_tokens, self.dim_input_vector))

        if len(words) > self.num_tokens - 1:
            words = words[len(words) - self.num_tokens + 1 : len(words)]
        elif len(words) < self.num_tokens - 1:
            words = ["<empty>"] * (self.num_tokens - 1 - len(words)) + words

        self.word_indices = []
        for i in range(self.num_tokens - 1):
            index = self.vocabulary.index(words[i]) if words[i] in self.vocabulary else -1
            self.word_indices.append(index)
            input_vectors[i, :] = self.embedding_matrix[index, :]

        empty_vector = self.embedding_matrix[-1, :]
        input_vectors[-1, :] = empty_vector
        # the vector to predict

        input_vectors = input_vectors + self.PE

        return input_vectors

    def unembed_train(self, output_vectors):
        self.output_vectors = output_vectors

        last_vector = output_vectors[-1, :]

        preactivations = np.matmul(last_vector, self.unembedding_matrix)
        preactivations = preactivations - np.max(preactivations)

        exps = np.exp(preactivations)
        prob_dist = exps * (1 / np.sum(exps))
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

    def backward_unembed_crossentropy(self, expected_last_word, LEARNING_RATE):
        index = self.vocabulary.index(expected_last_word) if expected_last_word in self.vocabulary else -1
        expected_prob_dist = np.ones((len(self.vocabulary))) * 0.0001
        expected_prob_dist[index] = 1 - len(self.vocabulary) * 0.0001

        dC_dP = np.divide(expected_prob_dist, np.clip(self.prob_dist, 0.001, None)) * -1

        dC_dpre = np.zeros((self.num_tokens, len(self.vocabulary)))
        dC_dpre[-1, :] = np.sum(np.diagflat(self.prob_dist) - np.outer(self.prob_dist * dC_dP, self.prob_dist), axis=0)

        dC_dUnembedding = np.matmul(np.transpose(self.output_vectors), dC_dpre)
        dC_dX = np.matmul(dC_dpre, np.transpose(self.unembedding_matrix))

        bruh = (1 / np.sum(dC_dUnembedding)) if np.sum(dC_dUnembedding) != 0 else 1
        self.unembedding_matrix = self.unembedding_matrix - dC_dUnembedding * LEARNING_RATE * abs(bruh)

        return dC_dX

    def backward_embed(self, dC_dY, LEARNING_RATE):
        for i in range(len(self.word_indices)):
            index = self.word_indices[i]
            bruh = (1 / np.sum(dC_dY[i, :])) if np.sum(dC_dY[i, :]) != 0 else 1
            self.embedding_matrix[index, :] = self.embedding_matrix[index, :] - dC_dY[i, :] * LEARNING_RATE * abs(bruh)
