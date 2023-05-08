import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class MoonDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = gnn.Sequential('x, edge_index, edge_weight', [
            (gnn.GraphConv(2, 20), 'x, edge_index, edge_weight -> x'),
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

        # IDEA maybe gradual graph pooling is more efficient and accurate
        self.linear = nn.Sequential(
            nn.Linear(396, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

        self.out = nn.Softmax()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        weights = data.edge_attr

        x = self.conv(x, edge_index, weights)
        x = x.view(-1, 198 * 2)
        x = self.linear(x)

        return F.softmax(x, dim = 1)
    
def main():
    from datasets import MoonClimbs
    from torch_geometric.loader import DataLoader

    dataset = MoonClimbs('data/moonclimbs', '6A+')
    loader = DataLoader(dataset, 5, shuffle = True)

    m = MoonDiscriminator()

    for x in loader:
        print(x)

        out = m(x)
        print(out)
        break

if __name__ == '__main__':
    main()