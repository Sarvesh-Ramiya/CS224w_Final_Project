from utils import fetch_protein_sequence
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import torch

with open('/Users/rbasto/Stanford projects/CS224W/general-set-2020-6-6-6_train_val.pkl', 'rb') as f:
  dataset = pickle.load(f)

sequences = {}
for i in tqdm(range(len(dataset))):
  dataset[i] = Data(**dataset[i].__dict__)  # allowing to use different pyg version
  dataset[i].x = dataset[i].x.to(torch.float32)
  seq = fetch_protein_sequence(dataset[i].pdbid[:4])
  if seq not in sequences.keys():
    sequences[seq] = []
  sequences[seq].append(dataset[i])

with open('sequences_data-6-6-6.pkl', 'wb') as file:
  pickle.dump(sequences, file)

count2 = 0
count3 = 0
for ids in sequences.values():
  if len(ids) > 3:
    count3 += 1
  elif len(ids) > 2:
    count2 += 1

print(count3)
print(count2)