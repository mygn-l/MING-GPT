from Multilayer_Feedforward import Multilayer_Feedforward
from Attention import Attention

class Transformer:
    def __init__(self, num_tokens, num_head, dim_head, dim_input_vector, dim_query, feedforward_dims):
        self.type = "transformer"

        self.attention = Attention(num_head, dim_head, dim_input_vector, dim_query)
        self.multilayer_feedforward = Multilayer_Feedforward(num_tokens, feedforward_dims)

    def forward_train(self, input_vectors):
        return self.multilayer_feedforward.forward_train(self.attention.forward_train(input_vectors))

    def backward(self, dC_dY, LEARNING_RATE):
        return self.attention.backward(self.multilayer_feedforward.backward(dC_dY, LEARNING_RATE), LEARNING_RATE)
