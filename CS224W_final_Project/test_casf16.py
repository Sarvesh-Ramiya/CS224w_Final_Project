import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pickle
from torch_geometric.data import Data, Batch
import sys
import layers as layers
import model as model
from tqdm import tqdm
import os
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'models'))
from models.fused_model import Net as fused
from models.graphLambda_model import Net as graphLambda
import numpy as np
import warnings
warnings.filterwarnings("ignore")

with open('general-set-2020-6-6-6_test.pkl', 'rb') as f:
        dataset = pickle.load(f)
for i in range(len(dataset)):
    dataset[i] = Data(**dataset[i].__dict__)  # allowing to use different pyg version
    dataset[i].x = dataset[i].x.to(torch.float32)

best_model = fused(dataset[0].num_node_features, dataset[0].num_edge_features, 64)
best_model.load_state_dict(torch.load("fused_MD_hdim_64_batch_16.pt",  map_location=torch.device('cpu')))
loss_function = torch.nn.MSELoss()

preds = []
truth = []

@torch.no_grad()
def test(model, loader):
    model.eval()
    loss_all = 0

    for data in tqdm(loader):
        data = data
        preds.append(model(data))
        truth.append(data.y)
        loss_all += loss_function(preds[-1], data.y).item() * len(data.y)
    return loss_all / len(loader.dataset)

val_loader = DataLoader(dataset, shuffle=True)

def pearson_r(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    top = torch.sum((x - mean_x) * (y - mean_y))
    bottom = torch.sqrt(torch.sum((x - mean_x) ** 2) * torch.sum((y - mean_y) ** 2))
    return top / bottom

print(np.sqrt(test(best_model, val_loader)))
print(pearson_r(torch.tensor(preds), torch.tensor(truth)))
