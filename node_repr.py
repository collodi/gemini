import json
import torch
import torch.nn.functional as F

grades = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

def get_climbs():
    with open('climbs.json') as f:
        return json.load(f)

def main():
    print('reading climbs')
    climbs = get_climbs()[:1]

    print('calculating node representations')
    nodes = torch.zeros((198, len(grades)), dtype=torch.float32)
    for climb in climbs:
        grade_i = grades.index(climb['grade'])
        for hold in climb['holds']:
            nodes[hold, grade_i] += 1.

    nodes = F.normalize(nodes, p = 1, dim = 1)

    print('exporting nodes.torch')
    torch.save(nodes, 'nodes.torch')

if __name__ == '__main__':
    main()