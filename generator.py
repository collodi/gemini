import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch

class ClimbGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.node_embedding = nn.Linear(20, 20)

        self.conv = gnn.Sequential('x, edge_index, edge_weight', [
            (gnn.GraphConv(20, 20), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(20, 10), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(10, 10), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(10, 5), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (gnn.GraphConv(5, 2), 'x, edge_index, edge_weight -> x'),
        ])

        self.linear = nn.Sequential(
            nn.Linear(396, 396),
            nn.ReLU(),
            nn.Linear(396, 396),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        weights = data.edge_attr

        x = self.node_embedding(x)
        x = self.conv(x, edge_index, weights)
        x = x.view(-1, 198 * 2) # [b_size * num_nodes, 2] -> [b_size, num_nodes * 2]
        x = self.linear(x)
        x = x.view(-1, 2) # [b_size, num_nodes * 2] -> [b_size * num_nodes, 2]
        # x = F.softmax(x, dim = 1)
        
        if type(data) is Data:
            return Data(x, edge_index, weights)
        
        return Batch(x, edge_index, weights, batch = data.batch, ptr = data.ptr)
    
def main():
    from template_graph import generate_template_graph, generate_random_template_graph

    nodes = torch.load('data/nodes.torch')
    edges = torch.load('data/distances.torch')
    grade = '6A+'
    reach = 70
    
    template = generate_template_graph(nodes, edges, grade, reach)
    graph = generate_random_template_graph(template)

    m = ClimbGenerator()
    climb = m(graph)

if __name__ == '__main__':
    main()