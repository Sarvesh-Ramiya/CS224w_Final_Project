import torch
import requests
import time
import torch_geometric.transforms as T

from pypdb import get_all_info
from Bio import SeqIO
from copy import deepcopy
from io import StringIO

transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
def remove_random_edges(graph, p):    
    graph = deepcopy(graph)
    num_edges = int(graph.edge_index.size()[1] / 2)
    keep_edge = (torch.rand(num_edges) > p).reshape(-1,1)
    keep_edge = torch.hstack((keep_edge, keep_edge)).flatten()
    graph.edge_index = graph.edge_index.T[keep_edge].T
    graph.edge_attr = graph.edge_attr[keep_edge]
    graph = transform(graph)
    return graph

def fetch_protein_sequence(pdb_id):
    """
    Fetch the protein sequence from the PDB database.
    """
    try:
        # Fetch FASTA sequence for the PDB ID
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
        response = requests.get(url)
        response.raise_for_status()
        # Parse the sequence from the response
        sequences = [record.seq for record in SeqIO.parse(StringIO(response.text), "fasta")]
        return str(sequences[0]) if sequences else None
    except Exception as e:
        print(f"Error fetching sequence for {pdb_id}: {e}")
        return None
    
