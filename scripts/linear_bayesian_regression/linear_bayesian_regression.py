import numpy as np


class LinearBayesianRegression():
    
    def __init__(self, n, alpha, beta):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.mu = np.array([0 for _ in range(n)]).T
        self.S = np.identity(n) / self.alpha
        return
    
    def learn(self, X, y):
        S = self.S + self.beta * np.dot(np.array(X).T, np.array(X))
        mu = self.beta * np.dot(S, np.dot(np.array(X).T, y)) + np.dot(self.S, self.mu)
        self.S = S
        self.mu = mu
        return
    
    def w_mostlike(self):
        return self.mu
    
    def w_sample(self, n=1):
        print(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]]))
        print(self.mu.T)
        print(self.S)
        return [np.random.multivariate_normal(self.mu.T, self.S).T for _ in range(n)]
    
    def w_distribute(self):
        return self.mu, self.S
    
    def y_mostlike(self, X):
        return np.dot(X, self.mu)
    
    def y_sample(self, X, n):
        return np.dot(X, self.w_sample(n))
    
    def y_distribute(self, X):
        return np.dot(X, self.mu), (1 / self.beta) + np.dot(np.array(X).T, np.dot(self.S, X))
    
    
    
    