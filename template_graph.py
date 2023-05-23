import torch
import torch.nn.functional as F

from torch_geometric.data import Data

EPSILON = 0.01
GRADES = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

def build_graph_nodes(nodes, grade):
    i = GRADES.index(grade)
    nodes = F.pad(nodes, (2, 2))
    return nodes[:, i:i+5]

def adjust_edge_weights(edges, reach):
    adjusted = torch.zeros_like(edges)
    nrows, ncols = edges.size()
    for i in range(nrows):
        for j in range(ncols):
            x = edges[i, j]
            adjusted[i, j] = -x / (2. * reach) + 1.

    return adjusted

def build_edges(edges, reach):
    edges = adjust_edge_weights(edges, reach)

    cnt = (edges > EPSILON).count_nonzero().div(2, rounding_mode = 'trunc')
    edge_index = torch.zeros((2, 2 * cnt), dtype = torch.long)
    weights = torch.zeros(2 * cnt, dtype = torch.float32)

    k = 0
    nrows, ncols = edges.size()
    for i in range(nrows):
        for j in range(ncols):
            if edges[i, j] < EPSILON:
                continue

            edge_index[0, k] = i
            edge_index[1, k] = j
            weights[k] = edges[i, j]
            k += 1

    return edge_index, weights

def generate_random_graph(template):
    x = torch.rand((198, 20), dtype = torch.float32)
    weights = torch.rand_like(template.edge_attr)
    return Data(x, template.edge_index, weights)

def generate_template_graph(nodes, edges, grade, reach):
    x = build_graph_nodes(nodes, grade)
    x = add_randomness(x)
    edge_index, weights = build_edges(edges, reach)
    return Data(x, edge_index, weights)

def add_randomness(tensor):
    return F.relu(torch.normal(tensor, tensor / 2))

def select_random_edges(graph):
    r = torch.rand_like(graph.edge_attr)
    p = r * graph.edge_attr
    threshold = (3 * p.max() + p.mean()) / 4
    return graph.edge_subgraph(p > threshold)

def main():
    from generator import ClimbGenerator
    from visualize import visualize_climb

    nodes = torch.load('data/nodes.torch')
    edges = torch.load('data/distances.torch')
    grade = '6A+'
    reach = 70

    template = generate_template_graph(nodes, edges, grade, reach)

    net = ClimbGenerator()
    
    for i in range(3):
        random_template = generate_random_graph(template)
        print(random_template.num_edges)

        climb = net(random_template)
        print(climb.x)
        visualize_climb(climb)

if __name__ == '__main__':
    main()