import torch

torch.set_printoptions(sci_mode=False)

def main():
    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
        [2, 3, 4, 3, 4, 0, 4, 0, 1, 0, 1, 2]
    ])

    x = torch.tensor([
        [0.0000, 0.0000, 0.0921, 0.0758, 0.1108],
        [0.0000, 0.0000, 0.1391, 0.0964, 0.1634],
        [0.0000, 0.0000, 0.0901, 0.0689, 0.1352],
        [0.0000, 0.0000, 0.2929, 0.1301, 0.1631],
        [0.0000, 0.0000, 0.0689, 0.0441, 0.0775]
    ])

    data = Data(x=x, edge_index=edge_index)
    print(data.is_undirected())
    return

    nodes = torch.load('data/nodes.torch')
    for i, x in enumerate(nodes):
        print(i)
        print_vector(x)

    print('===')

    xs = torch.load('data/randoms/processed/6A+/0.nodes')
    for x in xs:
        print_vector(x)

def print_vector(vec : torch.Tensor):
    for x in vec:
        print(f'{x.item():.6f}', end = ' ')

    print()

if __name__ == '__main__':
    main()