import os
import torch
import torch.nn.functional as F
from datasets import MoonClimbs
from generator import ClimbGenerator
from discriminator import MoonDiscriminator
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from template_graph import generate_template_graph, generate_random_graph
from visualize import visualize

torch.set_printoptions(sci_mode = False)

G_FN = 'models/generator.torch'
D_FN = 'models/discriminator.torch'

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nodes = torch.load('data/nodes.torch')
    edges = torch.load('data/distances.torch')
    grade = '6A+'
    reach = 70

    template_graph = generate_template_graph(nodes, edges, grade, reach)

    fixed_graph1 = generate_random_graph(template_graph).to(dev)
    fixed_graph2 = generate_random_graph(template_graph).to(dev)

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

    nepochs = 2
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
            graphs = [generate_random_graph(template_graph) for _ in range(b_size)]
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

if __name__ == '__main__':
    main()