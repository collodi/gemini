import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from template_graph import build_edges, add_randomness

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
        weights = add_randomness(weights)

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
    
if __name__ == '__main__':
    main()