import torch
import copy
import sys
import os
import random
import torch
import deepdish as dd
import torch.nn.functional as F
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'models'))
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from os.path import join
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, Dropout
from torch_geometric.nn import NNConv, Set2Set, GCNConv, global_add_pool, global_mean_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset #easily fits into cpu memory
from torch.utils.data import Subset
import pickle
import models.dynaformer_model as dynaformer_model
from utils import remove_random_edges
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'models'))
from models.graphLambda_model import Net
import models.dynaformer_model as dynaformer_model

writer = SummaryWriter(log_dir="logs/lambda_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

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

# Percentage of edges that have to removed to speed up training.
edge_dropout_rate = 0.2
batch_size = 32
train_ids, val_ids = train_test_split([i for i in range(len(dataset))], test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_ids), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_ids), batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_function = torch.nn.MSELoss()
def train(model, train_loader,epoch,device,optimizer):
    model.train()
    loss_all = 0
    batch_idx = 0
    for data in tqdm(train_loader):
        batch = data.to_data_list()
        # Removing random edges to accelerate training and increase expresiveness.
        for i in range(len(batch)):
            batch[i] = remove_random_edges(batch[i], edge_dropout_rate)
        data = Batch.from_data_list(batch)
        data.to_data_list()
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(data), data.y)
        writer.add_scalar("Train Loss (batch)", loss, (epoch - 1) * len(train_loader) + batch_idx)
        loss.backward()
        loss_all += loss.item() * len(data.y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        batch_idx += 1
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader,device):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += loss_function(model(data), data.y).item() * len(data.y)
        
    return loss_all / len(loader.dataset)


@torch.no_grad()
def test_predictions(model, loader):
    model.eval()
    pred = []
    true = []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
        true += data.y.detach().cpu().numpy().tolist()
    return pred, true

device = 'cpu'
best_val_error = None
best_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_errors, valid_errors,test_errors = [], [],[]
hidden_dim = 128
model = Net(dataset[0].num_node_features, hidden_dim).to(device)
lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                factor=0.5, patience=2,
                                min_lr=1e-6)

for epoch in range(1, 20):
    lr = scheduler.optimizer.param_groups[0]['lr']
    train_error = np.sqrt(train(model, train_loader,epoch,device,optimizer))
    writer.add_scalar("Train error (epoch)", train_error, epoch)
    val_error = np.sqrt(test(model, val_loader,device))
    writer.add_scalar("Val error (epoch)", val_error, epoch)
    valid_errors.append(val_error)

    if best_val_error is None or val_error <= best_val_error:
        best_val_error = val_error
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), 'model_checkpoints/graphLambda{MD_str}_hdim_{dim}_batch_{batch}.pt'.format(MD_str=MD_str, dim=hidden_dim, batch=batch_size))
    print('Epoch: {:03d}, LR: {:.7f}, Train RMSE: {:.7f}, Validation RMSE: {:.7f}'
        .format(epoch, lr, train_error, val_error))

print('leng of test errors = ', len(test_errors))
