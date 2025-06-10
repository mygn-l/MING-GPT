import numpy as np
import math

from Regularizer import Normalize_Factor, Dropout, KEEP_PROB
from Activation import Softmax, Softmax_derivative
from Adam import Adam

class Attention:
    def __init__(self, num_head, dim_head, dim_vector, dim_query):
        self.type = "attention"

        self.num_head = num_head
        self.dim_head = dim_head

        self.scaling_factor = 1 / math.sqrt(dim_query)

        self.heads = [
            [
                np.random.rand(dim_vector, dim_query) * 0.02 - 0.04, #query
                np.random.rand(dim_vector, dim_query) * 0.02 - 0.04, #key
                np.random.rand(dim_vector, dim_head) * 0.02 - 0.04 #value
            ]
            for i in range(num_head)
        ] #hope it's not pass-by-reference
        self.output_matrix = np.random.rand(dim_head * num_head, dim_vector) * 0.02 - 0.04

        self.query_m_prev = [np.zeros((dim_vector, dim_query)) for i in range(num_head)]
        self.query_v_prev = [np.zeros((dim_vector, dim_query)) for i in range(num_head)]
        self.key_m_prev = [np.zeros((dim_vector, dim_query)) for i in range(num_head)]
        self.key_v_prev = [np.zeros((dim_vector, dim_query)) for i in range(num_head)]
        self.value_m_prev = [np.zeros((dim_vector, dim_head)) for i in range(num_head)]
        self.value_v_prev = [np.zeros((dim_vector, dim_head)) for i in range(num_head)]

        self.Wo_m_prev = np.zeros((dim_head * num_head, dim_vector))
        self.Wo_v_prev = np.zeros((dim_head * num_head, dim_vector))

    def forward_train(self, input_vectors):
        self.input_vectors = input_vectors
        self.Q = []
        self.K = []
        self.V = []
        self.act = []

        single_head_outputs = []

        Wo = self.output_matrix
        for i in range(self.num_head):
            Wq = self.heads[i][0]
            Wk = self.heads[i][1]
            Wvd = self.heads[i][2]

            Q = np.matmul(input_vectors, Wq)
            K = np.matmul(input_vectors, Wk)
            V = np.matmul(input_vectors, Wvd)
            self.Q.append(Q)
            self.K.append(K)
            self.V.append(V)

            QKT = np.matmul(Q, np.transpose(K))

            pre = QKT * self.scaling_factor

            # softmax each vector
            act = np.zeros(pre.shape)
            for j in range(pre.shape[0]):
                pre[j, 0 : j+1] -= np.max(pre[j, 0 : j+1])
                act[j, 0 : j+1] = Softmax(pre[j, 0 : j+1])
            self.act.append(act)

            act = Dropout(act)

            ATTENTION = np.matmul(act, V)

            single_head_outputs.append(ATTENTION)

        heads_concat = np.concatenate(single_head_outputs, axis=1)
        self.heads_concat = heads_concat

        final_output = np.matmul(heads_concat, Wo)

        return final_output

    def backward(self, dC_dY, LEARNING_RATE):
        Wo = self.output_matrix
        dC_dHconcat = np.matmul(dC_dY, np.transpose(Wo))

        dC_dWo = np.matmul(np.transpose(self.heads_concat), dC_dY)

        dC_dX = np.zeros(dC_dY.shape)

        for i in range(self.num_head):
            Wq = self.heads[i][0]
            Wk = self.heads[i][1]
            Wvd = self.heads[i][2]

            dC_dHsplit = dC_dHconcat[:, i * self.dim_head : (i + 1) * self.dim_head]

            dC_dV = np.matmul(np.transpose(self.act[i]), dC_dHsplit)
            dC_dWv = np.matmul(np.transpose(self.input_vectors), dC_dV)

            dC_dact = np.matmul(dC_dHsplit, np.transpose(self.V[i])) / KEEP_PROB

            dC_dpre = np.empty(dC_dact.shape)
            for j in range(dC_dpre.shape[0]):
                dact_dpre = Softmax_derivative(self.act[i][j, :])
                dC_dpre[j, :] = np.sum(dact_dpre * dC_dact[j, :], axis=1)

            dC_dQKT = dC_dpre * self.scaling_factor

            dC_dQ = np.matmul(dC_dQKT, self.K[i])
            dC_dWq = np.matmul(np.transpose(self.input_vectors), dC_dQ)

            dC_dKT = np.matmul(np.transpose(self.Q[i]), dC_dQKT)
            dC_dK = np.transpose(dC_dKT)
            dC_dWk = np.matmul(np.transpose(self.input_vectors), dC_dK)

            # product rule
            dC_dX += np.matmul(dC_dV, np.transpose(Wvd))
            dC_dX += np.matmul(dC_dQ, np.transpose(Wq))
            dC_dX += np.matmul(dC_dK, np.transpose(Wk))

            # update only at the end
            query_adam = Adam(dC_dWq, self.query_m_prev[i], self.query_v_prev[i], LEARNING_RATE)
            self.heads[i][0] += query_adam[0]
            self.query_m_prev[i] = query_adam[1]
            self.query_v_prev[i] = query_adam[2]

            key_adam = Adam(dC_dWk, self.key_m_prev[i], self.key_v_prev[i], LEARNING_RATE)
            self.heads[i][1] += key_adam[0]
            self.key_m_prev[i] = key_adam[1]
            self.key_v_prev[i] = key_adam[2]

            value_adam = Adam(dC_dWv, self.value_m_prev[i], self.value_v_prev[i], LEARNING_RATE)
            self.heads[i][2] += value_adam[0]
            self.value_m_prev[i] = value_adam[1]
            self.value_v_prev[i] = value_adam[2]

        # update only at the end
        output_adam = Adam(dC_dWo, self.Wo_m_prev, self.Wo_v_prev, LEARNING_RATE)
        self.output_matrix += output_adam[0]
        self.Wo_m_prev = output_adam[1]
        self.Wo_v_prev = output_adam[2]

        dC_dX *= Normalize_Factor(dC_dX)
        #NORMALIZED

        return dC_dX
