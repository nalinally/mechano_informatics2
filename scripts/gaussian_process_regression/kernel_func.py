import numpy as np


def RBF(theta1, theta2):
    sigma = np.exp(theta1)
    l = np.exp(theta2)
    return lambda x1, x2: sigma**2 * np.exp(-np.linalg.norm(x1 - x2, ord=2)**2 / 2 / l**2)

def RBF_diff(theta1, theta2):
    sigma = np.exp(theta1)
    l = np.exp(theta2)
    X = lambda x1, x2: np.linalg.norm(x1 - x2, ord=2)**2 / 2 / l**2
    dkdsigma = lambda x1, x2: 2 * sigma * np.exp(-X(x1, x2))
    dkdl = lambda x1, x2: 2 * sigma**2 * X(x1, x2) / l * np.exp(-X(x1, x2))
    return lambda x1, x2: list([sigma * dkdsigma(x1, x2), l * dkdl(x1, x2)])
    