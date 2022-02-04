import torch
import torch.nn as nn
from model.layers.img_encoder import ResnetEncoder
from model.layers.sentence_encoder import make_encoder

class TripletMatch(nn.Module):
    def __init__(self, vec_dim=256, distance='l2_s', img_feats=(2, 4)):
        super(TripletMatch, self).__init__()
        self.vec_dim = vec_dim

        if distance == 'l2_s':
            self.dist_fn = lambda v1, v2: (v1 - v2).pow(2).sum(dim=-1)
        elif distance == 'l2':
            self.dist_fn = lambda v1, v2: (v1 - v2).pow(2).sum(dim=-1).sqrt()
        elif distance == 'cos':
            self.dist_fn = lambda v1, v2: 1.0 - nn.functional.cosine_similarity(v1, v2, dim=0)
        else:
            raise NotImplementedError

        self.resnet_encoder = ResnetEncoder(use_feats=img_feats)
        self.lang_embed = make_encoder()
        self.img_encoder = nn.Sequential(self.resnet_encoder, nn.Linear(self.resnet_encoder.out_dim, vec_dim))
        self.lang_encoder = nn.Sequential(self.lang_embed, nn.Linear(self.lang_embed.out_dim, vec_dim))
        return

    def forward(self, pos_imgs, pos_sents, neg_imgs=None, neg_sents=None):
        """
        :param pos_imgs: Tensor BxCxHxW
        :param neg_imgs: Tensor BxCxHxW
        :param pos_sents: list(str), len=B
        :param neg_sents: list(str), len=B
        """
        pos_img_vecs = self.img_encoder(pos_imgs)
        pos_sent_vecs = self.lang_encoder(pos_sents)
        pos_dist = self.dist_fn(pos_img_vecs, pos_sent_vecs)

        neg_img_loss = 0
        if neg_imgs is not None:
            neg_img_vecs = self.img_encoder(neg_imgs)
            neg_img_dist = self.dist_fn(pos_sent_vecs, neg_img_vecs)
            neg_img_losses = torch.relu(pos_dist - neg_img_dist + 1.0)
            neg_img_loss = torch.mean(neg_img_losses)

        neg_sent_loss = 0
        if neg_sents is not None:
            neg_sent_vecs = self.lang_encoder(neg_sents)
            neg_sent_dist = self.dist_fn(pos_img_vecs, neg_sent_vecs)
            neg_sent_losses = torch.relu(pos_dist - neg_sent_dist + 1.0)
            neg_sent_loss = torch.mean(neg_sent_losses)

        return neg_img_loss, neg_sent_loss