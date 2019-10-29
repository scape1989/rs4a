import numpy as np
import scipy as sp
import scipy.stats
import torch
import itertools
from matplotlib import pyplot as plt
from src.models import LogisticRegression
from src.smooth import *


def sim_data(n=100):
    rvs_1 = sp.stats.multivariate_normal([1, -2],[[2,0.3],[0.3,0.5]]).rvs(100)
    rvs_2 = sp.stats.multivariate_normal([-1, 2],[[2,0.3],[0.3,0.5]]).rvs(100)
    x = np.r_[rvs_1, rvs_2]
    y = np.r_[np.ones(100), np.zeros(100)]
    return x, y

if __name__ == "__main__":
    
    x, y = sim_data()

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    
    logreg = LogisticRegression(2)
    optimizer = torch.optim.Adam(logreg.parameters(), lr=0.01)

    for i in range(500):
        optimizer.zero_grad()
        loss = logreg.loss(x, y).mean()
        loss.backward()
        optimizer.step()
        print(f"{i}\tLoss: {loss.data:.2f}")

    length = 20
    x_axis = np.linspace(-10, 10, length)
    y_axis = np.linspace(-10, 10, length)
    
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    results_grid = np.zeros_like(x_grid)

    for i, j in itertools.product(range(length), range(length)):
        inputs = torch.tensor([x_grid[i,j], y_grid[i,j]], dtype=torch.float)
        results_grid[i,j] = logreg.forward(inputs).probs
        #results_grid[i,j] = smooth_predict_soft(logreg, inputs).probs

    plt.scatter(x[:100,0],x[:100,1])
    plt.scatter(x[100:,0],x[100:,1])
    plt.contourf(x_grid, y_grid, results_grid, alpha=0.3)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

