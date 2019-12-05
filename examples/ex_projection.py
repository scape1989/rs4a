import torch
from matplotlib import pyplot as plt
from src.attacks import project_onto_ball


if __name__ == "__main__":

    x = torch.rand((10, 2)) * 1.5 * torch.sign(torch.rand((10, 2)) - 0.5)
    x_proj = project_onto_ball(x, 1.0, p=1)

    origin = torch.tensor([0, 0])

    plt.quiver(*origin, x[:, 0], x[:, 1], scale=1, scale_units="xy", angles="xy", color="red", alpha=0.5)
    plt.quiver(*origin, x_proj[:, 0], x_proj[:, 1], scale=1, scale_units="xy", angles="xy", color="blue", alpha=0.5)
    plt.ylim((-2, 2))
    plt.xlim((-2, 2))

    plt.plot((0, 1), (1, 0), color="grey")
    plt.plot((1, 0), (0, -1), color="grey")
    plt.plot((0, -1), (-1, 0), color="grey")
    plt.plot((-1, 0), (0, 1), color="grey")

    plt.show()


