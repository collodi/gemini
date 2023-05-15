import os
import torch
import torch.nn.functional as F
from scipy.stats import norm
from datasets import MoonClimbs, RandomGraphs
from generator import ClimbGenerator
from discriminator import MoonDiscriminator
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from visualize_climb import visualize

torch.set_printoptions(sci_mode = False)

G_FN = 'models/generator.torch'
D_FN = 'models/discriminator.torch'

EPSILON = 0.01
GRADES = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nodes = torch.load('data/nodes.torch')
    edges = torch.load('data/distances.torch')
    grade = '6A+'
    reach = 70

    graph = generate_graph(nodes, edges, grade, reach)

    fixed_graph1 = random_graph_from_graph(graph).to(dev)
    fixed_graph2 = random_graph_from_graph(graph).to(dev)

    netG = load_generator().to(dev)
    optG = torch.optim.Adam(netG.parameters())

    netD = load_discriminator().to(dev)
    optD = torch.optim.Adam(netD.parameters())

    loss = torch.nn.BCELoss()

    real_dataset = MoonClimbs('data/moonclimbs', '6A+')
    real_loader = DataLoader(real_dataset, batch_size = 5, shuffle = True)

    climb = netG(fixed_graph1).detach().cpu()
    visualize(climb)

    climb = netG(fixed_graph2).detach().cpu()
    visualize(climb)

    nepochs = 3
    for epoch in range(nepochs):
        print(f'=== epoch {epoch}')

        for i, data in enumerate(real_loader):
            netD.zero_grad()

            # discriminator on real climbs
            x = data.to(dev)
            out = netD(x).view(-1)
            real_score = out.mean().item()

            b_size = out.size(0)
            label = torch.full((b_size, ), 1., dtype = torch.float32, device = dev)
            errD_real = loss(out, label)
            errD_real.backward()

            # discriminator on fake climbs
            graphs = [random_graph_from_graph(graph) for _ in range(b_size)]
            graphs = Batch.from_data_list(graphs).to(dev)

            fake = netG(graphs)
            out = netD(fake.detach()).view(-1)
            fake_score = 1. - out.mean().item()

            label.fill_(0.)
            errD_fake = loss(out, label)
            errD_fake.backward()

            optD.step()

            # generator
            netG.zero_grad()
            out = netD(fake).view(-1)
            gen_score = out.mean().item()

            label.fill_(1.)
            errG = loss(out, label)
            errG.backward()
            optG.step()

            # print intermediate results
            if i % 50 == 0:
                print(f'{i} / {len(real_loader)}')
                print(f'real score: {real_score:.6f}, real error: {errD_real:.6f}')
                print(f'fake score: {fake_score:.6f}, fake error: {errD_fake:.6f}')
                print(f'gen. score: {gen_score:.6f}, gen. error: {errG:.6f}')

        # save_generator(netG)
        # save_discriminator(netD)

        with torch.no_grad():
            climb = netG(fixed_graph1).detach().cpu()
            visualize(climb)

            climb = netG(fixed_graph2).detach().cpu()
            visualize(climb)

def generate_graph(nodes, edges, grade, reach):
    x = build_graph_nodes(nodes, grade)
    edge_index, weights = build_edges(edges, reach)
    return Data(x, edge_index, weights)

def random_graph_from_graph(graph):
    x, weights = graph.x, graph.edge_attr
    x = F.relu(torch.normal(x, x / 2))
    weights = F.relu(torch.normal(weights, weights/ 2))
    return Data(x, graph.edge_index, weights)

def save_generator(net):
    torch.save(net.state_dict(), G_FN)

def save_discriminator(net):
    torch.save(net.state_dict(), D_FN)

def load_generator():
    net = ClimbGenerator()
    if os.path.exists(G_FN):
        net.load_state_dict(torch.load(G_FN))

    return net

def load_discriminator():
    net = MoonDiscriminator()
    if os.path.exists(D_FN):
        net.load_state_dict(torch.load(D_FN))

    return net

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

if __name__ == '__main__':
    main()