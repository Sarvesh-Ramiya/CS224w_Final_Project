import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
from os.path import join
import random
import torch
import deepdish as dd
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, Dropout
from torch_geometric.nn import NNConv, Set2Set, GCNConv, global_add_pool, global_mean_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Subset
import pickle
import dynaformer_model as dynaformer_model
from utils import remove_random_edges
from torch_geometric.data import Data, Batch

"""
Citation: this code is largely based on the graphLambda official implementation at
https://github.com/i-Molecule/graphLambda
We made some Dynaformer-specific changes and some optimizations where necessary
"""

class Net(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim):
        super(Net, self).__init__()
        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.hidden_dim_middle = 4 * hidden_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_end = int(hidden_dim / 2)
        self.out_dims = [self.hidden_dim, 3*self.hidden_dim, self.hidden_dim_end, int(self.hidden_dim_end / 2)]
        
        #GCN-representation
        self.conv1 = GCNConv(self.input_dim_node, self.hidden_dim_middle, cached=False )
        self.bn01 = BatchNorm1d(self.hidden_dim_middle)
        self.conv2 = GCNConv(self.hidden_dim_middle, self.hidden_dim, cached=False )
        self.bn02 = BatchNorm1d(self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim, cached=False)
        self.bn03 = BatchNorm1d(self.out_dims[0])
        
        #GAT-representation
        self.gat1 = GATConv(self.input_dim_node, self.hidden_dim_middle,heads=3)
        self.bn11 = BatchNorm1d(self.hidden_dim_middle*3)
        self.gat2 = GATConv(self.hidden_dim_middle*3, self.hidden_dim,heads=3)
        self.bn12 = BatchNorm1d(self.hidden_dim*3)
        self.gat3 = GATConv(self.hidden_dim*3, self.hidden_dim,heads=3)
        self.bn13 = BatchNorm1d(self.out_dims[1])
        
        #GIN-representation
        fc_gin1=Sequential(Linear(self.input_dim_node, self.hidden_dim_middle), ReLU(), Linear(self.hidden_dim_middle, self.hidden_dim_middle))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(self.hidden_dim_middle)
        fc_gin2=Sequential(Linear(self.hidden_dim_middle, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(self.hidden_dim)
        fc_gin3=Sequential(Linear(self.hidden_dim, self.hidden_dim_end), ReLU(), Linear(self.hidden_dim_end, self.hidden_dim_end))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(self.out_dims[2])
        
        #Dynaformer representation
        self.dynaformer = dynaformer_model.DynaFormer(
            num_layers=2,
            input_node_dim=self.input_dim_node,
            node_dim=self.hidden_dim_middle,
            input_edge_dim=self.input_dim_edge,
            edge_dim=self.hidden_dim_middle,
            output_dim=self.out_dims[3],
            n_heads=16,
            ff_dim=self.hidden_dim_middle,
            max_in_degree=4,
            max_out_degree=4,
            max_path_distance=4,
            num_heads_spatial=4
        )
        
        #Fully connected layers for concatinating outputs
        self.fc1=Linear(sum(self.out_dims), self.hidden_dim)
        self.dropout1=Dropout(p=0.2,)
        self.fc2=Linear(self.hidden_dim, self.hidden_dim_end)
        self.dropout2=Dropout(p=0.2,)
        self.fc3=Linear(self.hidden_dim_end, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        y=x
        z=x
        
        #GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_mean_pool(x, data.batch)
        
        #GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_mean_pool(y, data.batch)
        
        #GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_mean_pool(z, data.batch)
        
        #Dynaformer
        w = global_mean_pool(self.dynaformer(data), data.batch)
        
        #Concatinating_representations
        cr=torch.cat((x,y,z,w),1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        cr = F.relu(cr).view(-1)
        return cr  
