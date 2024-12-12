import torch
import networkx as nx
import sys
import os
import pickle
import torch_geometric.transforms as T
import os
import numpy as np

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import Subset
from torch_geometric.utils.convert import to_networkx
from networkx import all_pairs_shortest_path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.nn.pool import global_mean_pool
from copy import deepcopy
from torch_geometric.utils import remove_isolated_nodes
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from copy import deepcopy
from utils import remove_random_edges

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'models'))
import models.dynaformer_layers as dynaformer_layers
import models.dynaformer_model as model

writer = SummaryWriter(log_dir="logs/dynaformer"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Flag controlling the type of dataset used.
# If set to true, Dataset with MD data is used, else data without MD simulations is used for training.
MD = True
if MD:
    MD_str = '_MD'
    dataset = []
    with open('md-refined2019-5-5-5_test.pkl', 'rb') as f:
        dataset_temp = pickle.load(f)
    for i in range(len(dataset_temp)):
        for j in range(len(dataset_temp[i])):
            dataset_temp[i][j] = Data(**dataset_temp[i][j].__dict__)  # allowing to use different pyg version
            dataset_temp[i][j].x = dataset_temp[i][j].x.to(torch.float32)
            dataset.append(dataset_temp[i][j])
    
else:
    MD_str = ''
    with open('refined-set-2020-5-5-5_train_val.pkl', 'rb') as f:
        dataset = pickle.load(f)
    for i in range(len(dataset)):
        dataset[i] = Data(**dataset[i].__dict__)  # allowing to use different pyg version
        dataset[i].x = dataset[i].x.to(torch.float32)

hidden_dim = 32
graph_model = model.DynaFormer(
    num_layers=2,
    input_node_dim=dataset[0].num_node_features,
    node_dim=hidden_dim,
    input_edge_dim=dataset[0].num_edge_features,
    edge_dim=hidden_dim,
    output_dim=1,
    n_heads=16,
    ff_dim=2 * hidden_dim,
    max_in_degree=4,
    max_out_degree=4,
    max_path_distance=4,
    num_heads_spatial=4
)

# Percentage of edges that have to removed to speed up training.
edge_dropout_rate = 0.9

batch_size = 8
train_ids, test_ids = train_test_split([i for i in range(len(dataset))], test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_ids), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(graph_model.parameters(), lr=1e-6)
loss_function = torch.nn.MSELoss()

best_val_error = None
for epoch in range(20):
    graph_model.train()
    batch_loss = 0.0
    batch_idx = 0
    for batch in tqdm(train_loader):
        batch = batch.to_data_list()
        for i in range(len(batch)):
            batch[i] = remove_random_edges(batch[i], edge_dropout_rate)
        batch = Batch.from_data_list(batch)
        batch.to_data_list()
        y = batch.y
        optimizer.zero_grad()
        output = global_mean_pool(graph_model(batch), batch.batch)
        loss = loss_function(output.flatten(), y.flatten())
        writer.add_scalar("Batch Loss", loss.item(), epoch * len(train_loader) + batch_idx)
        batch_loss += loss.item() * len(y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_model.parameters(), 1)
        optimizer.step()
        batch_idx += 1
    
    print("TRAIN_LOSS", batch_loss / len(train_loader.dataset))
    writer.add_scalar("Train Loss", batch_loss / len(train_loader.dataset), epoch)
    writer.add_scalar("Train Error", np.sqrt(batch_loss / len(train_loader.dataset)), epoch)
    graph_model.eval()
    batch_loss = 0.0
    for batch in tqdm(test_loader):
        y = batch.y
        with torch.no_grad():
            output = global_mean_pool(graph_model(batch), batch.batch)
            loss = loss_function(output.flatten(), y.flatten())

        batch_loss += loss.item() * len(y)

    val_loss = batch_loss / len(test_loader.dataset)
    print("EVAL LOSS", val_loss)
    writer.add_scalar("Eval Loss", val_loss, epoch)
    writer.add_scalar("Eval Error", np.sqrt(val_loss), epoch)

    if best_val_error is None or val_loss <= best_val_error:
        best_val_error = val_loss
        best_model = deepcopy(graph_model)
        torch.save(best_model.state_dict(), 'model_checkpoints/dynaformer{MD_str}_hdim_{dim}_batch_{batch}.pt'.format(MD_str=MD_str, dim=hidden_dim, batch=batch_size))
