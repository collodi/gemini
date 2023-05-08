import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Batch

class ClimbGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = gnn.Sequential('x, edge_index, edge_weight', [
            (gnn.GraphConv(6, 20), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(20, 15), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(15, 10), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(10, 5), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(5, 2), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
        ])

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        weights = data.edge_attr

        print(data.ptr)

        x = self.conv(x, edge_index, weights)
        x = F.softmax(x, dim = 1)
        return Batch(x, edge_index, weights, batch = data.batch, ptr = data.ptr)
    
def main():
    from datasets import RandomGraphs
    from torch_geometric.loader import DataLoader

    dataset = RandomGraphs('data/randoms', '6A+', 70, 100)
    loader = DataLoader(dataset, 5, shuffle = True)

    m = ClimbGenerator()

    for x in loader:
        print(x)

        out = m(x)
        print(out)
        break

if __name__ == '__main__':
    main()