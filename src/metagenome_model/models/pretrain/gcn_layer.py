#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : metagenome
# @File    : gcn_layer.py
# @Author  : zhangchao
# @Date    : 2024/8/28 10:30 
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGCNLayer(nn.Module):
    def __init__(self, output_dim, **kwargs):
        super().__init__()
        self.trans = nn.Linear(output_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.trans.weight)

    def forward(self, x, adj):
        """

        :param x: input features, shape (batch_size, num_nodes, input_dim)
        :param adj: adjacency matrix, shape (batch_size, num_nodes, num_nodes)

        process:
        1. calculate the degree matrix
        2. calculate the normalized adjacency matrix
        3. calculate the output features

            H(l+1) = D^-1/2 * A * D^-1/2 * H(l) * W

        :return: output features, shape (batch_size, num_nodes, output_dim)
        """
        bzs, num_nodes, _ = x.size()

        degree_matrix = torch.sum(adj, dim=-1, keepdim=False, dtype=x.dtype)
        degree_matrix = degree_matrix.pow(-0.5).reshape(bzs, num_nodes, 1)
        degree_matrix[degree_matrix == float('inf')] = 0
        degree_matrix = degree_matrix * torch.eye(num_nodes, dtype=x.dtype, device=x.device).unsqueeze(0)
        adj = (degree_matrix @ adj.to(x.dtype) @ degree_matrix)

        support = torch.einsum('ijk,ikl->ijl', adj, x)
        output = self.trans(support)
        return output


class Readout(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, adjacency, **kwargs):
        v_sum = adjacency @ x
        r_sum = adjacency.sum(-1)
        r_sum = r_sum[..., None].expand(v_sum.size())
        z = v_sum / r_sum
        z = torch.where(torch.isnan(z), torch.zeros_like(z), z)
        global_z = F.normalize(z, dim=-1)
        return torch.sigmoid(global_z)


class Discriminator(nn.Module):
    def __init__(self, input_dims, **kwargs):
        super().__init__()
        self.readout = Readout()
        self.disc = nn.Bilinear(input_dims, input_dims, 1)
        nn.init.xavier_uniform_(self.disc.weight.data)
        if self.disc.bias is not None:
            self.disc.bias.data.fill_(0.)

    def forward(self, pos_emb, neg_emb, adjacency, **kwargs):
        summary = self.readout(pos_emb, adjacency)
        pos_score = self.disc(pos_emb, summary)
        neg_score = self.disc(neg_emb, summary)
        logits = torch.cat((pos_score, neg_score), dim=-1)
        labels = F.one_hot(torch.zeros(
            (logits.size(0), logits.size(1)), dtype=torch.long, device=pos_emb.device), 2)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss

