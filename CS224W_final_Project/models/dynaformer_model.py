from typing import Union

import torch
from torch import nn
from torch_geometric.data import Data

from functional import shortest_path_distance, batched_shortest_path_distance
from dynaformer_layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding, EdgeEncoding

"""
Citation: this code is largely based on the Graphormer implementation at
https://github.com/leffff/graphormer-pyg
We made some Dynaformer-specific changes and some optimizations where necessary
"""

class DynaFormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 ff_dim: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int,
                 num_heads_spatial: int):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance
        self.num_heads_spatial = num_heads_spatial

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            num_heads=num_heads_spatial,
            embedding_size=self.node_dim
        )
        self.edge_encoding = EdgeEncoding(self.edge_dim, self.max_path_distance)

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if type(data) == Data:
            ptr = None
        else:
            ptr = data.ptr
        
        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)
        # Note: To Increase training speed Edge encoding is omitted. 
        edge_encoding = torch.zeros((x.shape[0], x.shape[0]))
        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, data.pos)
        ptr=None
        for layer in self.layers:
            x = layer(x, b, edge_encoding, ptr)
        x = self.node_out_lin(x)
        return x
