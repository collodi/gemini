import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tsne_torch import TorchTSNE as tSNE

torch.set_printoptions(sci_mode = False)

def visualize_climb(climb, fn = None):
    nrows = 18
    ncols = 11

    xs = [95, 150, 200, 250, 300, 350, 400, 455, 505, 560, 610]
    ys = [90, 140, 190, 240, 290, 340, 390, 450, 500, 550, 600, 655, 705, 755, 805, 860, 910, 960]

    nodes = climb.x
    fig, ax = plt.subplots()

    img = mpimg.imread('moon2019.jpg')
    ax.imshow(img)

    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            hold = nodes[i]
            hold_on = hold[1] > hold[0]

            if hold_on:
                x, y = xs[c], ys[r]
                circle = plt.Circle((x, y), radius = 20, color = 'b', linewidth = 3, fill = False)
                ax.add_patch(circle)

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)

def visualize_hold_embeddings():
    nodes = torch.load('data/nodes.torch')
    reduced = tSNE(initial_dims = 10, verbose = True).fit_transform(nodes)

    for i, pt in enumerate(reduced):
        plt.text(pt[0], pt[1], i, ha = 'center', va = 'center')

    plt.scatter(reduced[:, 0], reduced[:, 1], alpha = 0)
    plt.show()

def main():
    # climb = 'data/fakeclimbs/processed/6A+/70/1.climb'
    # climb = torch.load(climb)

    # visualize_climb(climb)

    visualize_hold_embeddings()

if __name__ == '__main__':
    main()