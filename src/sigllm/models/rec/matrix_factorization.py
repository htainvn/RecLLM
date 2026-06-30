import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)
        self._reset_parameters()

    def _reset_parameters(self):
        # Small-std init so the dot-product logits start near 0 (unsaturated).
        # Default nn.Embedding init is N(0,1); with embedding_size=256 that makes
        # logits ~N(0,256) (std~16), saturating BCEWithLogitsLoss -> AUC stuck ~0.5.
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        with torch.no_grad():
            self.user_embedding.weight[self.padding_index].zero_()
            self.item_embedding.weight[self.padding_index].zero_()

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