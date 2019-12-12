import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":

    y = np.load("./ckpts/cifar_gaussian_05/labels.npy")
    direct = np.load("./ckpts/cifar_gaussian_05_direct/preds_smooth.npy")
    augment = np.load("./ckpts/cifar_gaussian_05/preds_smooth.npy")

    preds_direct = direct[np.arange(10000), y.astype(int)]
    preds_augment = augment[np.arange(10000), y.astype(int)]

    top_1_direct = np.argmax(direct, axis=1)
    top_1_augment = np.argmax(augment, axis=1)

    print(-np.log(preds_direct + 1e-4).mean())
    print(-np.log(preds_augment + 1e-4).mean())

    print(-np.log(preds_direct[top_1_direct == y] + 1e-4).mean())
    print(-np.log(preds_augment[top_1_augment == y] + 1e-4).mean())

    plt.hist(preds_direct, alpha=0.5)
    plt.hist(preds_augment, alpha=0.5)
    plt.show()

