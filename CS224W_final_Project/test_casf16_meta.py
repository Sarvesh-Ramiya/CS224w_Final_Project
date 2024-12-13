import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from networkx import all_pairs_shortest_path
from sklearn.model_selection import train_test_split
import pickle
from torch_geometric.data import Data, Batch
import sys
import layers as layers
import model as model
from tqdm import tqdm
from torch_geometric.nn.pool import global_mean_pool
from models.graphLambda_model import Net
import higher
import random

best_model = Net(9, 64)
best_model.load_state_dict(torch.load("/model_checkpoints/meta_lambda_MD_ils_1_obs_8_hd_64_2024-12-11_10-18-13.pt"))

optimizer = torch.optim.AdamW(best_model.parameters(), lr=1e-3)
inner_opt = torch.optim.Adam(best_model.parameters(), lr=1e-3)
preds = []
truth = []
loss_function = torch.nn.MSELoss()
def inner_loop(model, loss, inner_opt, task, num_inner_loop_steps, it):
    #MY CONTRIBUTION
    """
    Idea: make a copy of the model, update parameters of the copied model for num_inner_loop_steps

    Input: model (instance of the NN), inner_opt (optimizer for inner loop), task (Batch object with support and query set)
    Returns: loss on query after adaptation (it's a 2D tensor, output of GeoDiff)
    """
    for i in range(len(task)):
        support = Batch.from_data_list(random.sample(task[:i] + task[i+1:], 3))  #needs to be a batch object with only support set
        query = Batch.from_data_list([task[i]])  #also a batch object with only query set
        inner_opt.zero_grad()

        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            query_loss_array = []
            for _ in range(num_inner_loop_steps):
                support_loss = loss(model(support), support.y)
                diffopt.step(support_loss)
        preds.append(model(query))
        truth.append(query.y)
        query_loss = loss(model(query), query.y)
        return query_loss

with open('sequences_data-6-6-6_test.pkl', 'rb') as f:
    dataset_temp = pickle.load(f)

dataset = []
for data in dataset_temp.values():
    if len(data) >= 4:
        dataset.append(data)

val_set = dataset

def validate(it):
    best_model.train()
    #MY CONTRIBUTION
    num_inner_loop_steps = 3
    val_loss = []
    for i, task in enumerate(tqdm(val_set, desc='Validation')):
        # batch = batch.to(args.device)
        query_loss = inner_loop(best_model, loss_function, inner_opt, task, num_inner_loop_steps, it) #batch here is just one task
        val_loss.append(query_loss.detach())
    val_loss = torch.mean(torch.tensor(val_loss))
    val_error = torch.mean(torch.sqrt(val_loss))

    # writer.add_scalar('val/loss (epoch)', val_loss, it)
    # writer.add_scalar('val/error (epoch)', val_error, it)
    # writer.flush()
    return val_loss, val_error

def pearson_r(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    top = torch.sum((x - mean_x) * (y - mean_y))
    bottom = torch.sqrt(torch.sum((x - mean_x) ** 2) * torch.sum((y - mean_y) ** 2))
    return top / bottom

print(validate(0))
print(pearson_r(torch.tensor(preds), torch.tensor(truth)))
