from typing import Tuple

import torch
from torch import nn
from torch_geometric.utils import degree
import numpy as np

"""
Citation: this code is largely based on the Graphormer implementation at
https://github.com/leffff/graphormer-pyg
We made some Dynaformer-specific changes and some optimizations where necessary
"""

def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(),
                                          self.max_in_degree - 1)
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(),
                                           self.max_out_degree - 1)

        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x
    
# this spatial encoding with gaussians is Dynaformer-specific, and differs from Graphormer    
class SpatialEncoding(nn.Module):  
    def __init__(self, num_heads: int, embedding_size: int):
        """
        :param num_heads: number of encoding heads in the GBF function
        :param embedding_size: dimension of node embedding vector
        """
        super().__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        
        self.means = nn.Parameter(torch.randn(self.num_heads))
        self.stds = nn.Parameter(torch.randn(self.num_heads))
        self.weights_dist = nn.Parameter(torch.randn(2 * self.embedding_size + 1))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        :param x: node feature matrix (feature vectors in rows)
        :param coords: spatial coordinates of atoms (N x 3)
        :return: torch.Tensor, spatial Encoding matrix (N x N)
        """
        
        norms = (torch.linalg.vector_norm(coords, ord=2, dim=1) ** 2).reshape(-1,1)
        distances = torch.sqrt(torch.clamp(norms - 2 * coords @ coords.T + norms.T, min=0))
        x1 = x.unsqueeze(1)
        x0 = x.unsqueeze(0)
        N, D = x.shape
        concats = torch.cat((x1.expand(N, N, D), distances.unsqueeze(-1), x0.expand(N, N, D)), dim=-1)
        concats = concats.reshape(N ** 2, 2 * D + 1) @ self.weights_dist
        spatial_matrix = concats.reshape(N, N)
        min = spatial_matrix.min()
        max = spatial_matrix.max()
        spatial_matrix = (spatial_matrix - min) / (max - min)
        stds = self.stds.reshape(-1,1,1)
        a = (2*np.pi) ** 0.5
        spatial_matrix = (1 / (a * stds)) * torch.exp(-0.5 * (spatial_matrix - self.means.reshape(-1, 1, 1)) ** 2 / (stds ** 2))
        spatial_matrix = torch.mean(spatial_matrix, dim=0)  # mean pooling
        return spatial_matrix

class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_weights = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
        cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
        weights_inds = torch.arange(0, self.max_path_distance)
        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                cij[src][dst] = (self.edge_weights[weights_inds[:len(path_ij)]] * edge_attr[path_ij]).sum(dim=1).mean()

        cij = torch.nan_to_num(cij)
        return cij

class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        """
        super().__init__()

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                edge_encoding,
                ptr=None) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param b: spatial Encoding matrix
        :param edge_encoding: edge encodings
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(
             next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)


        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = edge_encoding
        a = self.compute_a(key, query, ptr)
        a = (a + b + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5

        return a


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                edge_encoding,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, b, edge_encoding, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, ff_dim, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
        )
        self.ln_1 = nn.LayerNorm(self.node_dim)
        self.ln_2 = nn.LayerNorm(self.node_dim)
        self.ff = nn.Sequential(
                    nn.Linear(self.node_dim, self.ff_dim),
                    nn.GELU(),
                    nn.Linear(self.ff_dim, self.node_dim)
        )


    def forward(self,
                x: torch.Tensor,
                b: torch,
                edge_encoding,
                ptr) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), b, edge_encoding, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new
