from Multilayer_Feedforward import Multilayer_Feedforward
from Attention import Attention
from Layer_Normalization import Layer_Normalization

class Transformer:
    def __init__(self, num_tokens, num_head, dim_head, dim_input_vector, dim_query, feedforward_dims):
        self.type = "transformer"

        self.layer_normalizer1 = Layer_Normalization(num_tokens)
        self.attention = Attention(num_head, dim_head, dim_input_vector, dim_query)
        self.layer_normalizer2 = Layer_Normalization(num_tokens)
        self.multilayer_feedforward = Multilayer_Feedforward(num_tokens, feedforward_dims)

    def forward_train(self, input_vectors):
        attentioned = self.attention.forward_train(self.layer_normalizer1.forward_train(input_vectors))
        # Don't need to add attentioned to input_vectors, since attention already internally adds
        feedforwarded = self.multilayer_feedforward.forward_train(self.layer_normalizer2.forward_train(attentioned))
        return attentioned + feedforwarded

    def backward(self, dC_dY):
        dC_dZ = self.layer_normalizer2.backward(self.multilayer_feedforward.backward(dC_dY) + dC_dY)
        dC_dX = self.layer_normalizer1.backward(self.attention.backward(dC_dZ))
        return dC_dX
