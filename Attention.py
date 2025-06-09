import numpy as np
import math

class Attention:
    def __init__(self, num_head, dim_head, dim_vector, dim_query):
        self.type = "attention"

        self.num_head = num_head
        self.dim_head = dim_head

        self.scaling_factor = 1 / math.sqrt(dim_query)

        self.heads = [
            [
                np.random.rand(dim_vector, dim_query), #query
                np.random.rand(dim_vector, dim_query), #key
                np.random.rand(dim_vector, dim_head) #value
            ]
            for i in range(num_head)
        ] #hope it's not pass-by-reference
        self.output_matrix = np.random.rand(dim_head * num_head, dim_vector)

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
            act = np.empty(pre.shape)
            for j in range(pre.shape[0]):
                pre[j, 0 : j+1] = pre[j, 0 : j+1] - np.max(pre[j, 0 : j+1])
                exps = np.exp(pre[j, 0 : j+1])
                act[j, 0 : j+1] = exps * (1 / np.sum(exps))
                act[j, j+1 :] = 0
            self.act.append(act)

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

            dC_dact = np.matmul(dC_dHsplit, np.transpose(self.V[i]))

            dC_dpre = np.empty(dC_dact.shape)
            for j in range(dC_dpre.shape[0]):
                act_vector = self.act[i][j, :]
                dC_dpre[j, :] = np.sum(np.diagflat(act_vector) - np.outer(act_vector * dC_dact[j, :], act_vector), axis=0)

            dC_dQKT = dC_dpre * self.scaling_factor

            dC_dQ = np.matmul(dC_dQKT, self.K[i])
            dC_dWq = np.matmul(np.transpose(self.input_vectors), dC_dQ)

            dC_dKT = np.matmul(np.transpose(self.Q[i]), dC_dQKT)
            dC_dK = np.transpose(dC_dKT)
            dC_dWk = np.matmul(np.transpose(self.input_vectors), dC_dK)

            # product rule
            dC_dX = dC_dX + np.matmul(dC_dV, np.transpose(Wvd))
            dC_dX = dC_dX + np.matmul(dC_dQ, np.transpose(Wq))
            dC_dX = dC_dX + np.matmul(dC_dK, np.transpose(Wk))

            # update at the end
            bruh = (1 / np.sum(dC_dWq)) if np.sum(dC_dWq) != 0 else 1
            self.heads[i][0] = self.heads[i][0] - dC_dWq * LEARNING_RATE * abs(bruh)
            bruh = (1 / np.sum(dC_dWk)) if np.sum(dC_dWk) != 0 else 1
            self.heads[i][1] = self.heads[i][1] - dC_dWk * LEARNING_RATE * abs(bruh)
            bruh = (1 / np.sum(dC_dWv)) if np.sum(dC_dWv) != 0 else 1
            self.heads[i][2] = self.heads[i][2] - dC_dWv * LEARNING_RATE * abs(bruh)

        # update at the end
        bruh = (1 / np.sum(dC_dWo)) if np.sum(dC_dWo) != 0 else 1
        self.output_matrix = self.output_matrix - dC_dWo * LEARNING_RATE * abs(bruh)

        bruh = (1 / np.sum(dC_dX)) if np.sum(dC_dX) != 0 else 1
        dC_dX = dC_dX * bruh
        #NORMALIZED

        return dC_dX
