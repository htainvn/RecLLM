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
        # Init scale matters a lot here. Default nn.Embedding init is N(0,1) -> with
        # embedding_size=256 the dot-product logits are ~N(0,256) (std~16), saturating
        # BCEWithLogitsLoss -> AUC ~0.5. But std=0.01 is the opposite failure: logits
        # start ~0.0016, so the model must grow embedding norms ~25x before it can
        # separate classes, which takes hundreds of epochs at lr=1e-3 (each of the
        # ~34k item rows is touched only ~21x/epoch). std=0.1 puts the initial logit
        # std at ~0.16 -- unsaturated AND already at a usable scale.
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
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