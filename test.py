import torch

torch.set_printoptions(sci_mode=False)

def main():
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
        print(f'{x.item():6f}', end = ' ')

    print()

if __name__ == '__main__':
    main()