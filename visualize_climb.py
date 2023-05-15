import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

torch.set_printoptions(sci_mode = False)

nrows = 18
ncols = 11

xs = [95, 150, 200, 250, 300, 350, 400, 455, 505, 560, 610]
ys = [90, 140, 190, 240, 290, 340, 390, 450, 500, 550, 600, 655, 705, 755, 805, 860, 910, 960]

def visualize(climb, fn = None):
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

def main():
    climb = 'data/fakeclimbs/processed/6A+/70/1.climb'
    #climb = 'data/moonclimbs/processed/6A+/0.climb'
    climb = torch.load(climb)

    visualize(climb)

if __name__ == '__main__':
    main()