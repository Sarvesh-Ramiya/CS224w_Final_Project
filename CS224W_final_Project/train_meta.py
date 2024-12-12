import os
import shutil
import argparse
from tqdm.auto import tqdm
import torch
import sys
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'models'))
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
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
parser.add_argument('--resume_iter', type=str, default=None) #here I changed type to str to allow for the named checkpoints
parser.add_argument('--train_set',type=str, default="drugs")
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--maml', type=int, default=1)
parser.add_argument('--val_freq', type=float, default=1) 
parser.add_argument('--val_size', type=float, default=1) #do on a subset of validation set
parser.add_argument('--max_iter', type=int, default=10050)
parser.add_argument('--inner_loop_steps', type=int, default=1)
parser.add_argument('--outer_batch_size', type=int, default=16)
parser.add_argument('--curriculum', type=int, default=0)
parser.add_argument('--learn_inner_lr', type=int, default=0)
parser.add_argument('--scale_ilr', type=int, default=1)
parser.add_argument('--scale_olr', type=int, default=1)
parser.add_argument('--sup_size', type=int, default=3)
parser.add_argument('--query_size', type=int, default=1)
parser.add_argument('--decrease_sup_size', type=int, default=0)
parser.add_argument('--msl', type=int, default=0)
parser.add_argument('--curriculum_type', type=int, default=0)
args = parser.parse_args()

writer = SummaryWriter(log_dir="logs/metaLambda_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Datasets and loaders

with open('/Users/rbasto/Stanford projects/CS224W/sequences_data-6-6-6.pkl', 'rb') as f:
  dataset_temp = pickle.load(f)

dataset = []
for data in dataset_temp.values():
    if len(data) >= 4:
        dataset.append(data)

train_ids, val_ids = train_test_split([i for i in range(len(dataset))], test_size=0.2, random_state=42)
train_set = Subset(dataset, train_ids)
val_set = Subset(dataset, val_ids)

model = graphLambda(dataset[0][0].num_node_features, 64)
# model = fused(dataset[0][0].num_node_features, dataset[0][0].num_edge_features, 32)
lr_outer = 1e-3
lr_inner = 1e-3

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer)
inner_opt = torch.optim.Adam(model.parameters(), lr=lr_inner)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                 factor=0.5, patience=1,
#                                 min_lr=1e-6, threshold=0.2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_function = torch.nn.MSELoss()

num_examples_per_task = args.sup_size
p = 0.9
def get_per_step_loss_importance_vector(current_step):
    """
    Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
    loss towards the optimization loss.
    :return: A tensor to be used to compute the weighted average of the loss, useful for
    the MSL (Multi Step Loss) mechanism.
    """
    loss_weights = np.ones(shape=(args.inner_loop_steps)) * (
            1.0 / args.inner_loop_steps)
    reference_step_number = int(len(dataset) / args.outer_batch_size)
    decay_rate = 1.0 / args.inner_loop_steps / reference_step_number
    min_value_for_non_final_losses = 0.03 / args.inner_loop_steps
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (current_step * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(
        loss_weights[-1] + (current_step * (args.inner_loop_steps - 1) * decay_rate),
        1.0 - ((args.inner_loop_steps - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    loss_weights = torch.Tensor(loss_weights) #.to(device=args.device)
    return loss_weights

def inner_loop(model, loss, inner_opt, task, num_inner_loop_steps, it):
    #MY CONTRIBUTION
    """
    Idea: make a copy of the model, update parameters of the copied model for num_inner_loop_steps

    Input: model (instance of the NN), inner_opt (optimizer for inner loop), task (Batch object with support and query set)
    Returns: loss on query after adaptation (it's a 2D tensor, output of GeoDiff)
    """
    # data_list = task.to_data_list()
    support = Batch.from_data_list(random.sample(task, args.sup_size))  #needs to be a batch object with only support set
    query = Batch.from_data_list(random.sample(task, args.query_size))  #also a batch object with only query set
    inner_opt.zero_grad()

    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
        if args.learn_inner_lr == True:
            for name, g in zip(fnet.param_names, diffopt.param_groups): #have to make sure both param_names and param_groups etc in same order
                g['lr'] = fnet.inner_lrs[name]
        query_loss_array = []
        for _ in range(num_inner_loop_steps):
            support_loss = loss(model(support), support.y)
            diffopt.step(support_loss)
            if args.msl == True:
                query_loss = loss(model(query), query.y)
                query_loss_array.append(query_loss)
        
    if args.msl == True:
        weights = get_per_step_loss_importance_vector(it)
        query_loss = weights[0]*query_loss_array[0]
        for i in range(len(query_loss_array)-1):
            query_loss += weights[i+1]*query_loss_array[i+1]
        return query_loss
    else:
        query_loss = loss(model(query), query.y)
        return query_loss

def train(it):
    model.train()
    optimizer.zero_grad()
    num_inner_loop_steps = args.inner_loop_steps #experiment with this
    batch_size_outer_loop = args.outer_batch_size #experiment with this
        
    # if MAML == True:
    #MY CONTRIBUTION
    #compute inner loop each time, and then compute outer loop
    #in inner loop adapt parameters manually, then do loss.backward for outer loop
    outer_loss_batch = []
    task_batch = deepcopy(random.sample(list(train_set), args.outer_batch_size))
    train_error = []
    for task in task_batch:
        for i in range(len(task)):
            task[i] = remove_random_edges(task[i], p)
        query_loss = inner_loop(model, loss_function, inner_opt, task, num_inner_loop_steps, it)
        train_error.append(torch.sqrt(query_loss).mean())
        query_loss = query_loss.mean()
        query_loss.backward()
        outer_loss_batch.append(query_loss.detach())
    loss = torch.mean(torch.stack(outer_loss_batch))
    orig_grad_norm = clip_grad_norm_(model.parameters(), 1)      
    optimizer.step()      

    writer.add_scalar('train/loss (batch)', loss, it)
    writer.add_scalar('train/error (batch)', torch.mean(torch.tensor(train_error)), it)
    writer.flush()

    return loss

def validate(it):
    model.train()
    #MY CONTRIBUTION
    num_inner_loop_steps = args.inner_loop_steps #experiment with this
    val_loss = []
    for i, task in enumerate(tqdm(val_set, desc='Validation')):
        query_loss = inner_loop(model, loss_function, inner_opt, task, num_inner_loop_steps, it) #batch here is just one task
        val_loss.append(query_loss.detach())
    val_loss = torch.mean(torch.tensor(val_loss))
    val_error = torch.mean(torch.sqrt(val_loss))

    writer.add_scalar('val/loss (epoch)', val_loss, it)
    writer.add_scalar('val/error (epoch)', val_error, it)
    writer.flush()
    return val_loss, val_error

try:
    init_sup_size = args.sup_size
    min_sup_size = 0
    max_iter = args.max_iter
    # freq_task_change = int(size_train_MAML*epoch_size/(4)) #this is aligned with MAML/GeoDiff split
    done = False
    for epoch in range(1, 1001):
        avg_train_loss = []
        num_steps = int(len(train_set) / args.outer_batch_size)
        for it in tqdm(range(num_steps)):  
            avg_train_loss.append(train((epoch - 1) * num_steps + it))
        avg_train_loss = torch.mean(torch.tensor(avg_train_loss))
        val_loss, val_error = validate(epoch)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation RMSE: {:.7f}'
        .format(epoch, lr, avg_train_loss, val_error))
except KeyboardInterrupt:
    print("Terminating...")

