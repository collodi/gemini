import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from template_graph import adjust_edge_weights

EPSILON = 0.01
GRADES = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

# .climb contains data for discriminator
# each node is represented by a softmax(len 2 vector) [off, on]

def get_climbs():
    with open('data/climbs.json') as f:
        return json.load(f)

def get_grade_counts():
    climbs = get_climbs()
    climbs = [x for x in climbs if x['method'] == 'Feet follow hands']
    return [sum(1 for x in climbs if x['grade'] == grade) for grade in GRADES]

def get_edges(edges, holds):
    n = len(holds)
    cnt = n * (n - 1)

    edge_idx = torch.zeros((2, cnt), dtype = torch.long)
    weights = torch.zeros(cnt, dtype = torch.float32)

    k = 0
    for i, h1 in enumerate(holds):
        for j, h2 in enumerate(holds):
            if i == j:
                continue

            edge_idx[0, k] = i
            edge_idx[1, k] = j
            weights[k] = edges[h1, h2]
            k += 1

    return edge_idx, weights

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

        nodes = torch.load('data/nodes.torch')

        edges = torch.load('data/distances.torch')
        edges = adjust_edge_weights(edges, self.reach)

        grade_dir = os.path.join(self.processed_dir, self.grade)
        if not os.path.exists(grade_dir):
            os.makedirs(grade_dir)

        for i, climb in enumerate(climbs):
            holds = climb['holds']

            climb_nodes = nodes[holds]
            edge_idx, weights = get_edges(edges, holds)

            data = Data(climb_nodes, edge_idx, weights, climb = climb)
            torch.save(data, os.path.join(grade_dir, f'{i}.climb'))

    def len(self):
        return self.num
    
    def get(self, i):
        fn = os.path.join(self.processed_dir, self.grade, f'{i}.climb')
        return torch.load(fn)
    
def main():
    moons = MoonClimbs('data/moonclimbs', '6A+')
    
if __name__ == '__main__':
    main()