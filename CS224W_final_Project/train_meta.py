import os
import shutil
import argparse
from tqdm.auto import tqdm
import torch
import sys
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'models'))
import torch.utils.tensorboard
from torch_geometric.data import Batch
from torch.utils.data import Subset
from copy import deepcopy
import higher
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random
from models.graphLambda_model import Net as graphLambda
from models.fused_model import Net as fused
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import remove_random_edges

parser = argparse.ArgumentParser()
# parser.add_argument('config', type=str)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--train_set',type=str, default="drugs")
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--max_iter', type=int, default=10050)
parser.add_argument('--inner_loop_steps', type=int, default=1)
parser.add_argument('--outer_batch_size', type=int, default=16)
parser.add_argument('--sup_size', type=int, default=3)
parser.add_argument('--query_size', type=int, default=1)
parser.add_argument('--decrease_sup_size', type=int, default=0)  # whether to apply the curriculum
parser.add_argument('--hidden_dim', type=int, default=64)  
parser.add_argument('--MD', type=int, default=0)
args = parser.parse_args()

MD = args.MD
# use different dataset depending on whether using MD data or not
if MD:
    MD_str = '_MD'
    with open('sequences_data_MD-5-5-5.pkl', 'rb') as f:
        dataset_temp = pickle.load(f)
else:
    MD_str = ''
    with open('sequences_data-6-6-6.pkl', 'rb') as f:
        dataset_temp = pickle.load(f)


dataset = []
for data in dataset_temp.values():
    if len(data) >= 4 and not MD:  # only include the proteins with at least for examples
        dataset.append(data)
    elif MD:  # for MD already have at least around 100 examples per task
        dataset.append(data)

# do a train/val split
train_ids, val_ids = train_test_split([i for i in range(len(dataset))], test_size=0.2)
train_set = Subset(dataset, train_ids)
val_set = Subset(dataset, val_ids)

# which model we are applying meta-learning to
hidden_dim = args.hidden_dim
model_str = "_lambda"
if model_str == "_lambda":
    model = graphLambda(dataset[0][0].num_node_features, hidden_dim)
elif model_str == "_fused":
    model = fused(dataset[0][0].num_node_features, dataset[0][0].num_edge_features, hidden_dim)

signature = "_ils_{ils}_obs_{obs}_hd_{hd}".format(ils=args.inner_loop_steps, obs=args.outer_batch_size, hd=hidden_dim)
dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(log_dir="logs/meta{model_str}{MD_str}{sig}_".format(model_str=model_str, MD_str=MD_str, sig=signature)+dt)

lr_outer = 1e-3
lr_inner = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer)
inner_opt = torch.optim.Adam(model.parameters(), lr=lr_inner)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                 factor=0.5, patience=1,
#                                 min_lr=1e-6, threshold=0.2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_function = torch.nn.MSELoss()
p = 0.9  # fraction of edges to drop during training

def inner_loop(model, loss, inner_opt, task, num_inner_loop_steps, it):
    """
    Input: model, inner_opt (optimizer for inner loop), task (Batch object with support and query set)
    Returns: loss on query after adaptation (it's a 2D tensor, output of GeoDiff)
    
    Makes a copy of the model, update parameters of the copied model for num_inner_loop_steps
    
    """
    support = Batch.from_data_list(random.sample(task, args.sup_size))  # batch object with support set
    query = Batch.from_data_list(random.sample(task, args.query_size))  # batch object with query set
    inner_opt.zero_grad()
    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
        for _ in range(num_inner_loop_steps):
            support_loss = loss(model(support), support.y)
            diffopt.step(support_loss)
    query_loss = loss(model(query), query.y)
    return query_loss

def train(it):
    model.train()
    optimizer.zero_grad()
    outer_loss_batch = []
    task_batch = deepcopy(random.sample(list(train_set), args.outer_batch_size))
    train_error = []
    # for each task in the batch, perform inner loop updates, aggregate those losses, 
    # and update global parameters afterwards
    # Follows MAML algorithm
    for task in task_batch:
        for i in range(len(task)):
            task[i] = remove_random_edges(task[i], p)
        query_loss = inner_loop(model, loss_function, inner_opt, task, args.inner_loop_steps, it)
        train_error.append(query_loss.mean())
        query_loss = query_loss.mean()
        query_loss.backward()
        outer_loss_batch.append(query_loss.detach())
    loss = torch.mean(torch.stack(outer_loss_batch))
    optimizer.step()
    #print(model.inner_lrs)        

    writer.add_scalar('train/loss (batch)', loss, it)
    writer.add_scalar('train/error (batch)', torch.sqrt(loss), it)
    writer.flush()

    return loss

def validate(it):
    model.train()
    val_loss = []
    # here we do inner loop updates for each task, then compute loss based on adapted parameters
    for i, task in enumerate(tqdm(val_set, desc='Validation')):
        query_loss = inner_loop(model, loss_function, inner_opt, task, args.inner_loop_steps, it) #batch here is just one task
        val_loss.append(query_loss.detach())
    val_loss = torch.mean(torch.tensor(val_loss))
    val_error = torch.mean(torch.sqrt(val_loss))

    writer.add_scalar('val/loss (epoch)', val_loss, it)
    writer.add_scalar('val/error (epoch)', val_error, it)
    writer.flush()
    return val_loss, val_error

try:
    min_sup_size = 1
    max_query_size = 3
    freq_task_change = 10
    best_val_error = None
    for epoch in range(1, 1001):
        avg_train_loss = []
        num_steps = int(len(train_set) / args.outer_batch_size)
        # run train iteration over num_step batches
        for it in tqdm(range(num_steps)):
            avg_train_loss.append(train((epoch - 1) * num_steps + it))
        avg_train_loss = torch.mean(torch.tensor(avg_train_loss))
        avg_train_error = torch.sqrt(avg_train_loss)
        writer.add_scalar('train/loss (epoch)', avg_train_loss, epoch)
        writer.add_scalar('train/error (epoch)', avg_train_error, epoch)
    
        val_loss, val_error = validate(epoch)

        # if we are decreasing the support set size (curriculum)
        if (args.decrease_sup_size == True) and (epoch % freq_task_change == 0):
            args.sup_size = max(min_sup_size, args.sup_size - 1)
            args.query_size = min(max_query_size, args.query_size + 1)
        
        lr = optimizer.param_groups[0]['lr']
        print('Epoch: {:03d}, LR: {:.7f}, Train Loss: {:.7f}, Train RMSE: {:.7f}, Val RMSE: {:.7f}'
        .format(epoch, lr, avg_train_loss, avg_train_error, val_error))
        # print(args.sup_size)
        # print(args.query_size)

        # save the best model
        if best_val_error is None or val_error <= best_val_error:
            best_val_error = val_error
            best_model = deepcopy(model)
            torch.save(best_model.state_dict(), 'model_checkpoints/meta{model_str}{MD_str}{sig}_{dt}.pt'.format(model_str=model_str, MD_str=MD_str, sig=signature, dt=dt))
    
except KeyboardInterrupt:
    print("Terminating...")

