import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from scipy.stats import norm

EPSILON = 0.01
GRADES = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

# .climb contains data for discriminator
# each node is represented by a softmax(len 2 vector) [off, on]

# .graph contains data for generator
# each node is represented by 5 numbers for grade distribution and 1 number for randomness

def get_climbs():
    with open('data/climbs.json') as f:
        return json.load(f)
    

def get_grade_counts():
    climbs = get_climbs()
    climbs = [x for x in climbs if x['method'] == 'Feet follow hands']
    return [sum(1 for x in climbs if x['grade'] == grade) for grade in GRADES]

def ensure_elem_larger(x, i):
    j = (i + 1) % 2
    if x[i] >= x[j]:
        return x.clone()
    
    return x.flip([0])

def build_climb_nodes(climb):
    nodes = torch.rand(198, 2)
    # maybe using softmax will make discriminator perform better?
    nodes = F.normalize(nodes, p = 1, dim = 1)

    holds = set(climb['holds'])
    for i in range(198):
        mx_i = 1 if i in holds else 0
        nodes[i] = ensure_elem_larger(nodes[i], mx_i)

    return nodes

def adjust_edge_weights(edges, reach):
    D = norm(loc = reach / 2, scale = 7.875)

    adjusted = torch.zeros_like(edges)
    nrows, ncols = edges.size()
    for i in range(nrows):
        for j in range(ncols):
            v = edges[i, j]
            adjusted[i, j] = D.cdf(v + 2) - D.cdf(v - 2)

    return adjusted

def build_edges(edges, reach):
    edges = adjust_edge_weights(edges, reach)

    cnt = (edges > EPSILON).count_nonzero().div(2, rounding_mode = 'trunc')
    edge_index = torch.zeros((2, cnt), dtype = torch.long)
    weights = torch.zeros(cnt, dtype = torch.float32)

    k = 0
    nrows, ncols = edges.size()
    for i in range(nrows):
        for j in range(i, ncols):
            if edges[i, j] < EPSILON:
                continue

            edge_index[0, k] = i
            edge_index[1, k] = j
            weights[k] = edges[i, j]
            k += 1

    return edge_index, weights

def build_graph_nodes(nodes, grade):
    i = GRADES.index(grade)
    nodes = F.pad(nodes, (2, 2))
    return nodes[:, i:i+5]

class RandomGraphs(Dataset):
    def __init__(self, root, grade, reach, num):
        self.grade = grade
        self.reach = reach
        self.num = num

        super().__init__(root)
    
    @property
    def processed_file_names(self):
        return [os.path.join(self.grade, f'{i}.graph') for i in range(self.num)]
    
    def process(self):
        nodes = torch.load('data/nodes.torch')
        grade_nodes = build_graph_nodes(nodes, self.grade)

        edges = torch.load('data/distances.torch')
        edge_index, weights = build_edges(edges, self.reach)

        grade_dir = os.path.join(self.processed_dir, self.grade)
        if not os.path.exists(grade_dir):
            os.makedirs(grade_dir)

        for i in range(self.num):
            r = torch.rand((nodes.size()[0], 1))
            graph_nodes_random = torch.cat((grade_nodes, r), dim = 1)

            data = Data(graph_nodes_random, edge_index, weights)
            torch.save(data, os.path.join(grade_dir, f'{i}.graph'))

    def len(self):
        return self.num
    
    def get(self, i):
        return torch.load(os.path.join(self.processed_dir, self.grade, f'{i}.graph'))
    
class MoonClimbs(Dataset):
    def __init__(self, root, grade):
        self.grade = grade
        self.reach = 70 # inches

        grade_i = GRADES.index(grade)
        self.num = get_grade_counts()[grade_i]

        super().__init__(root)
    
    @property
    def processed_file_names(self):
        return [os.path.join(self.grade, f'{i}.climb') for i in range(self.num)]
    
    def process(self):
        climbs = get_climbs()
        climbs = [x for x in climbs if x['method'] == 'Feet follow hands']
        climbs = [x for x in climbs if x['grade'] == self.grade]

        edges = torch.load('data/distances.torch')
        edge_index, weights = build_edges(edges, self.reach)

        grade_dir = os.path.join(self.processed_dir, self.grade)
        if not os.path.exists(grade_dir):
            os.makedirs(grade_dir)

        for i, climb in enumerate(climbs):
            nodes = build_climb_nodes(climb)
            data = Data(nodes, edge_index, weights, climb = climb)
            torch.save(data, os.path.join(grade_dir, f'{i}.climb'))
    
    def len(self):
        return self.num
    
    def get(self, i):
        return torch.load(os.path.join(self.processed_dir, self.grade, f'{i}.climb'))
    
def main():
    moon6aplus = MoonClimbs('data/moonclimbs', '6A+')
    
if __name__ == '__main__':
    main()