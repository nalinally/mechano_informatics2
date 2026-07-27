import numpy as np


def RBF(sigma, l):
    def f(x1, x2):
        return sigma**2 * np.exp(-np.linalg.norm(x1 - x2, ord=2)**2 / 2 / l**2)
    return f

def RBF_diff(sigma, l):
    def f(x1, x2):
        X = np.linalg.norm(x1 - x2, ord=2)**2 / 2 / l**2
        y1 = 2 * sigma * np.exp(-X)
        y2 = 2 * sigma**2 * X / l * np.exp(-X)
        return list([y1, y2])
    return f
    