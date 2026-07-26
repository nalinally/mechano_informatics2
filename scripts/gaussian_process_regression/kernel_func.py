import numpy as np


def RBF(sigma, l):
    theta1 = sigma**2
    theta2 = 2 * l**2
    def f(x1, x2):
        return theta1 * np.exp(-np.linalg.norm(x1 - x2, ord=2)**2 / theta2)
    return f

def RBF_diff(sigma, l):
    theta1 = sigma**2
    theta2 = 2 * l**2
    def f(x1, x2):
        X = np.linalg.norm(x1 - x2, ord=2)**2 / theta2
        y1 = np.exp(-X)
        y2 = theta1 / theta2 * X * np.exp(-X)
        return list([y1, y2])
    return f
    