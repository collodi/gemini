import torch
from torch import linalg as LA

def hold_num(hold):
    r, c = hold
    return r * 11 + c

def cross_row(r1, r2, row):
    mn = min(r1, r2)
    mx = max(r1, r2)

    return mn <= row and row <= mx

def calc_dist(h1, h2):
    r1, c1 = h1
    r2, c2 = h2

    dx = abs(c1 - c2) * 7.875
    dy = abs(r1 - r2) * 7.875

    if cross_row(r1, r2, 5.5):
        dy += 0.75

    if cross_row(r1, r2, 11.5):
        dy += 0.75

    return LA.norm(torch.tensor([dx, dy]))

def main():
    print('calculating hold distances')
    edges = torch.zeros((198, 198), dtype = torch.float32)
    holds = [(r, c) for c in range(11) for r in range(18)]

    for h1 in holds:
        for h2 in holds:
            n1 = hold_num(h1)
            n2 = hold_num(h2)

            edges[n1, n2] = calc_dist(h1, h2)

    print('exporting distances.torch')
    torch.save(edges.detach(), 'data/distances.torch')

if __name__ == '__main__':
    main()