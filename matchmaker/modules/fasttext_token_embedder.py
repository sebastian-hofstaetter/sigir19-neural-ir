from torch.nn.modules.sparse import EmbeddingBag
import numpy as np
import torch


class FastTextEmbeddingBag(EmbeddingBag):

    # embedding_matrix = fasttext input_matrix -> exported by generate_fasttext_weights.py (and loaded via numpy.load)
    def __init__(self, embedding_matrix, sparse=False):
        embedding_matrix_shape = embedding_matrix.shape
        super().__init__(embedding_matrix_shape[0], embedding_matrix_shape[1], sparse=sparse)
        self.weight.data.copy_(torch.FloatTensor(embedding_matrix))

    def get_output_dim(self):
        return self.weight.shape[1]

    # token_subwordIds is created via module/fasttext_token_embedder 
    # -> the ids must be created from the same model as the embedding_matrix via generate_fast_text_vocab_mapping.py
    def forward(self, token_subwordIds):

        #token_subwordIds.shape = batch, token_max (padded), subword_max (padded)

        #one_view.shape = batch * token_max (padded), subword_max (padded)
        one_view = token_subwordIds.view(-1, token_subwordIds.shape[2])

        #out.shape = batch * token_max (padded), embedding_dim
        out = super().forward(one_view)

        #out_batched.shape = batch, token_max (padded), embedding_dim
        out_batched = out.view(token_subwordIds.shape[0], token_subwordIds.shape[1], -1)

        return out_batched
