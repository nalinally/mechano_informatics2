import numpy as np


def RBF(theta1, theta2):
    def f(x1, x2):
        return theta1 * np.exp(-np.dot(x1 - x2, x1 - x2) / 2 / theta2)
    return f

def optimize_hiper_param(X, y, n, k, dkdtheta):
    def dLdtheta(theta):
        K = [[k(x, x_) for x in X] for x_ in X]
        K_inv = np.linalg.inv(K)
        dKdtheta = [[dkdtheta(x, x_) for x in X] for x_ in X]
        print(dKdtheta.shape)
        print(K.shape)
        return [-np.trace(np.dot(K_inv, dKdtheta[:, :, i])) for i in range(dKdtheta[0][0])]
    