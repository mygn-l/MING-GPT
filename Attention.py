import numpy as np
import math

from Regularizer import Dropout
from Activation import Softmax, Softmax_derivative
from Adam import Adam
from config import KEEP_PROB

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
        ]
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

            Q = input_vectors @ Wq
            K = input_vectors @ Wk
            V = input_vectors @ Wvd
            self.Q.append(Q)
            self.K.append(K)
            self.V.append(V)

            QKT = Q @ K.T

            pre = QKT * self.scaling_factor

            # softmax each vector
            act = Softmax(pre)
            self.act.append(act)

            ATTENTION = Dropout(act) @ V

            single_head_outputs.append(ATTENTION)

        heads_concat = np.concatenate(single_head_outputs, axis=1)
        self.heads_concat = heads_concat

        final_output = heads_concat @ Wo

        return final_output + input_vectors

    def backward(self, dC_dY):
        Wo = self.output_matrix
        dC_dHconcat = dC_dY @ Wo.T

        dC_dWo = self.heads_concat.T @ dC_dY

        dC_dX = np.ones(dC_dY.shape)

        for i in range(self.num_head):
            Wq = self.heads[i][0]
            Wk = self.heads[i][1]
            Wvd = self.heads[i][2]

            dC_dHsplit = dC_dHconcat[:, i * self.dim_head : (i + 1) * self.dim_head]

            dC_dV = self.act[i].T @ dC_dHsplit
            dC_dWv = self.input_vectors.T @ dC_dV

            dC_dact = dC_dHsplit @ self.V[i].T / KEEP_PROB

            dC_dpre = np.empty(dC_dact.shape)
            for j in range(dC_dpre.shape[0]):
                dact_dpre = Softmax_derivative(self.act[i][j, :])
                dC_dpre[j, :] = dact_dpre @ dC_dact[j, :]

            dC_dQKT = dC_dpre * self.scaling_factor

            dC_dQ = dC_dQKT @ self.K[i]
            dC_dWq = self.input_vectors.T @ dC_dQ

            dC_dK = dC_dQKT.T @ self.Q[i]
            dC_dWk = self.input_vectors.T @ dC_dK

            # product rule
            dC_dX += dC_dV @ Wvd.T
            dC_dX += dC_dQ @ Wq.T
            dC_dX += dC_dK @ Wk.T

            # update only at the end
            query_adam = Adam(dC_dWq, self.query_m_prev[i], self.query_v_prev[i])
            self.heads[i][0] += query_adam[0]
            self.query_m_prev[i] = query_adam[1]
            self.query_v_prev[i] = query_adam[2]

            key_adam = Adam(dC_dWk, self.key_m_prev[i], self.key_v_prev[i])
            self.heads[i][1] += key_adam[0]
            self.key_m_prev[i] = key_adam[1]
            self.key_v_prev[i] = key_adam[2]

            value_adam = Adam(dC_dWv, self.value_m_prev[i], self.value_v_prev[i])
            self.heads[i][2] += value_adam[0]
            self.value_m_prev[i] = value_adam[1]
            self.value_v_prev[i] = value_adam[2]

        # update only at the end
        output_adam = Adam(dC_dWo, self.Wo_m_prev, self.Wo_v_prev)
        self.output_matrix += output_adam[0]
        self.Wo_m_prev = output_adam[1]
        self.Wo_v_prev = output_adam[2]

        return dC_dX
