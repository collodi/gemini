import torch

torch.set_printoptions(sci_mode=False)

def main():
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data

    nodes = torch.load('data/nodes.torch')
    for i, x in enumerate(nodes):
        print(f'1, {i} -> {torch.dot(nodes[1], x).item():.6f}')

    print('===')

def print_vector(vec : torch.Tensor):
    for x in vec:
        print(f'{x.item():.6f}', end = ' ')

    print()

if __name__ == '__main__':
    main()