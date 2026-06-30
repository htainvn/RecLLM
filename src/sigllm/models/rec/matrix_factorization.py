import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)
    
    def user_encoder(self,user_ids):
        return self.user_embedding(user_ids)

    def item_encoder(self,item_ids):
        return self.item_embedding(item_ids)

    def compute(self):
        return None, None

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        matching = torch.mul(user_embeddings, item_embeddings).sum(dim=-1)
        return matching