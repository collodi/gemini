import json
import torch
import torch.nn as nn
import torch.nn.functional as F

grades = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

hold_cnt = 198
climb_cnt = 24227

def get_climbs():
    with open('data/climbs.json') as f:
        return json.load(f)

class HoldEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.h_embed = nn.Embedding(hold_cnt, 20)
        self.c_embed = nn.Embedding(climb_cnt, 20)

    def forward(self, holds, climb):
        hold_vectors = self.h_embed(holds)
        hold_vectors = F.normalize(hold_vectors)

        climb_vector = self.c_embed(climb)
        climb_vector = F.normalize(climb_vector)

        dots = torch.mm(climb_vector, hold_vectors.t())
        return F.sigmoid(dots)

def export_nodes(m):
    holds = torch.arange(198, dtype = torch.long)
    nodes = F.normalize(m.h_embed(holds))
    torch.save(nodes, 'data/nodes.torch')

def holds_in_climb(holds, climb, all_climbs):
    out = [(h in all_climbs[climb]['holds']) for h in holds]
    out = torch.tensor(out, dtype = torch.float32)
    return out

def holds_in_climbs(holds, climbs, all_climbs):
    out = [holds_in_climb(holds, c, all_climbs) for c in climbs]
    return torch.vstack(out)

def train(m, opt, loss, b_size, climbs):
    holds = torch.arange(198, dtype = torch.long)

    for i in range(climb_cnt // b_size):
        r_climbs = torch.randint(climb_cnt, (b_size, ))
        y = holds_in_climbs(holds, r_climbs, climbs)

        opt.zero_grad()

        out = m(holds, r_climbs)
        err = loss(out, y)
        err.backward()

        opt.step()

        if i % 250 == 0:
            print(f'{i * b_size} / {climb_cnt}: {err.item():.6f}')


def main():
    print('reading climbs')
    climbs = get_climbs()

    print('learning hold embeddings')
    m = HoldEmbedding()
    opt = torch.optim.Adam(m.parameters())
    loss = nn.BCELoss()

    b_size = 1
    n_epochs = 1000

    for epoch in range(n_epochs):
        print(f'=== epoch {epoch}')
        train(m, opt, loss, b_size, climbs)

        print('exporting nodes.torch')
        export_nodes(m)

if __name__ == '__main__':
    main()