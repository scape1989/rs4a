#
# toy example to understand estimators of the cross-entropy loss of the smooth classifier
#
#   -log E[f(x + d)], d ~ N(0, 1)
#
# here we let f(x) = x^2 since it has closed-form ground truth, and compare estimators:
#
# 1. gaussian data augmentation, where we estimate
#
#   E[-log f(x + d)] >= -log E[f(x + d)] due to Jensen's inequality
#
# 2. plug-in estimator, where we directly take Monte Carlo samples of the goal
#
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return x ** 2

def truth(x):
    return -np.log(1 + x ** 2)

if __name__ == "__main__":

    x = np.linspace(-0.3, 0.3, 100)[:, np.newaxis]
    samples = np.random.randn(100, 50)
    
    plug_in = -np.log(f(samples + x).mean(axis=1))
    gda = -np.log(f(samples + x)).mean(axis=1) # high variance! also upper bound is too loose
    truth = truth(x)
    
    plt.plot(x, plug_in, label="plug_in")
    plt.plot(x, gda, label="gda")
    plt.plot(x, truth, label="truth")

    plt.legend()
    plt.show()

