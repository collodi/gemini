import os
import json
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from scipy.stats import norm
from generator import ClimbGenerator

EPSILON = 0.01
GRADES = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

# .nodes contains data for generator
# each node is represented by 5 numbers for grade distribution and 1 number for randomness
# this file is flexible for loading the dataset with a custom reach parameter

# .climb contains data for discriminator
# each node is represented by a softmax(len 2 vector) [off, on]

# .graph contains data for generator
# basically .nodes + edges (given reach)

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
    # IDEA maybe using softmax will make discriminator perform better?
    nodes = F.normalize(nodes, p = 1, dim = 1)

    holds = set(climb['holds'])
    for i in range(198):
        mx_i = 1 if i in holds else 0 # [off, on]
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

class FakeClimbs(Dataset):
    def __init__(self, root, generator, grade, reach, *, delete_old = False):
        self.grade = grade
        self.reach = reach

        grade_i = GRADES.index(grade)
        self.num = get_grade_counts()[grade_i]

        if delete_old:
            for fn in self.processed_file_names:
                fn = os.path.join(root, 'processed', fn)
                if os.path.exists(fn):
                    os.remove(fn)

        self.generator = generator or ClimbGenerator()
        super().__init__(root)

    @property
    def processed_file_names(self):
        return [os.path.join(self.grade, str(self.reach), f'{i}.climb') for i in range(self.num)]

    def process(self):
        graphs = RandomGraphs('data/randoms', self.grade, self.reach, self.num)

        category_dir = os.path.join(self.processed_dir, self.grade, str(self.reach))
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        for i, graph in enumerate(graphs):
            climb = self.generator(graph)
            torch.save(climb, os.path.join(category_dir, f'{i}.climb'))

    def len(self):
        return self.num
    
    def get(self, i):
        fn = os.path.join(self.processed_dir, self.grade, str(self.reach), f'{i}.climb')
        return torch.load(fn)

class RandomGraphs(Dataset):
    def __init__(self, root, grade, reach, num):
        self.grade = grade
        self.reach = reach
        self.num = num

        edges = torch.load('data/distances.torch')
        self.edge_index, self.weights = build_edges(edges, self.reach)

        super().__init__(root)
    
    @property
    def processed_file_names(self):
        return [os.path.join(self.grade, f'{i}.nodes') for i in range(self.num)]
    
    def process(self):
        nodes = torch.load('data/nodes.torch')
        grade_nodes = build_graph_nodes(nodes, self.grade)

        grade_dir = os.path.join(self.processed_dir, self.grade)
        if not os.path.exists(grade_dir):
            os.makedirs(grade_dir)

        for i in range(self.num):
            r = torch.rand((nodes.size()[0], 1))
            graph_nodes_random = torch.cat((grade_nodes, r), dim = 1)
            torch.save(graph_nodes_random, os.path.join(grade_dir, f'{i}.nodes'))

    def len(self):
        return self.num
    
    def get(self, i):
        nodes = torch.load(os.path.join(self.processed_dir, self.grade, f'{i}.nodes'))
        return Data(nodes, self.edge_index, self.weights)
    
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
        fn = os.path.join(self.processed_dir, self.grade, f'{i}.climb')
        return torch.load(fn)
    
def main():
    moons = MoonClimbs('data/moonclimbs', '6A+')
    fakes = FakeClimbs('data/fakeclimbs', None, '6A+', 70, delete_old = True)
    
if __name__ == '__main__':
    main()