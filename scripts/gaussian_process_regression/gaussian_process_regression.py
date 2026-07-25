import numpy as np

class GaussianProcessRegression():
    
    def __init__(self, k):
        self.k = k
        self.K = None
        self.K_inv = None
        self.X = None
        self.y = None

    def learn(self, X, y):
        self.K = [[self.k(x, x_) for x_ in X] for x in X]
        self.K_inv = np.linalg.inv(self.K)
        self.X = X
        self.y = y
        
    def y_dist(self, X):
        kN = [[self.k(x, x_) for x_ in self.X] for x in X]
        K_input = [[self.k(x, x_) for x_ in X] for x in X]
        mu = np.dot(kN, np.dot(self.K_inv, self.y))
        S = K_input - np.dot(kN, np.dot(self.K_inv, kN.T))
        return mu, S
    
    def y(self, X):
        mu, _ = self.y_dist(X)
        return mu
                

        
        