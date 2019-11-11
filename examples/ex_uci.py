import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models import ForecastNN
from src.attacks import fgsm_attack, pgd_attack
from src.smooth import *
from src.noises import *


dataset_name_to_loader = {
    "housing": lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, delim_whitespace=True),
    "concrete": lambda: pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"),
    "wine": lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=";"),
    "kin8nm": lambda: pd.read_csv("data/uci/kin8nm.csv"),
    "naval": lambda: pd.read_csv("data/uci/naval-propulsion.txt", delim_whitespace=True, header=None).iloc[:,:-1],
    "power": lambda: pd.read_excel("data/uci/power-plant.xlsx"),
    "energy": lambda: pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx").iloc[:,:-1],
    "protein": lambda: pd.read_csv("data/uci/protein.csv")[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'RMSD']],
    "yacht": lambda: pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data", header=None, delim_whitespace=True),
    "msd": lambda: pd.read_csv("data/uci/YearPredictionMSD.txt").iloc[:, ::-1],
}


if __name__ == "__main__":

    np.random.seed(1)
    
    argparser =  ArgumentParser()
    argparser.add_argument("--dataset", default="housing", type=str)
    argparser.add_argument("--iterations", default=5000, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    noise = GaussianNoise(0.5, "cpu")

    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values[:,np.newaxis]
    print(f"Shape: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)

    model = ForecastNN(X_train.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for i in range(args.iterations):
        optim.zero_grad()
        loss = model.loss(X_train, y_train).mean()
        loss.backward()
        optim.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\tLoss: {loss.data:.2f}")
    
    X_adv = pgd_attack(model, X_val, y_val, 0.1, steps=40, clamp=(-5, 5))
    adv_loss = -model.forecast(model.forward(X_adv)).log_prob(y_val).mean()

    print(f"ORIG: {loss.data:.2f}")
    print(f"ADV: {adv_loss.data:.2f}")

    lower = y_train.mean() - 2.5 * y_train.std()
    upper = y_train.mean() + 2.5 * y_train.std()
    axis = torch.tensor(np.linspace(lower, upper, 200), dtype=torch.float)
    pdf_adv = model.forecast(model.forward(X_adv)).log_prob(axis)
    pdf_val = model.forecast(model.forward(X_val)).log_prob(axis)
    pdf_smooth = smooth_predict_soft(model, X_adv, noise, 256, clamp=(-5, 5)).log_prob(axis)

    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(axis, torch.exp(pdf_val[i,:]).data.numpy(), "-", 
                 color="black", label="Original")
        plt.plot(axis, torch.exp(pdf_adv[i,:]).data.numpy(), "--", 
                 color="red", label="Adversarial")
        plt.plot(axis, torch.exp(pdf_smooth[i,:]).data.numpy(), "--", 
                 color="blue", label="Smoothed")
        plt.legend()

    plt.tight_layout()
    plt.show()

