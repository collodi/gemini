import os
import json
import base64
from scipy.stats import norm
import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch_geometric.data import Data

def get_climbs():
    with open('data/climbs.json') as f:
        return json.load(f)
    
def ensure_elem_larger(x, i):
    j = (i + 1) % 2
    if x[i] >= x[j]:
        return x.clone()
    
    return x.flip([0])
    
def build_nodes(climb):
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

def build_edges(edges, eps):
    cnt = (edges > eps).count_nonzero() // 2
    edge_index = torch.zeros((2, cnt), dtype = torch.long)
    weights = torch.zeros(cnt, dtype = torch.float32)

    k = 0
    nrows, ncols = edges.size()
    for i in range(nrows):
        for j in range(i, ncols):
            if edges[i, j] < eps:
                continue

            edge_index[0, k] = i
            edge_index[1, k] = j
            weights[k] = edges[i, j]
            k += 1

    return edge_index, weights
    
def main():
    climbs = get_climbs()
    climbs = [x for x in climbs if x['method'] == 'Feet follow hands']

    edges = torch.load('data/distances.torch')

    reach = 70 # inches
    edges = adjust_edge_weights(edges, reach)
    edge_index, weights = build_edges(edges, 0.01)

    for climb in climbs:
        nodes = build_nodes(climb)
        graph = Data(x = nodes, edge_index = edge_index, edge_attr = weights, climb = climb)

        folder = 'data/truths/' + climb['grade']
        fn = climb['name'] + ' by ' + climb['setBy']
        fn = base64.urlsafe_b64encode(fn.encode('utf-8')).decode('utf-8')

        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(graph, f'{folder}/{fn}.graph')

if __name__ == '__main__':
    main()