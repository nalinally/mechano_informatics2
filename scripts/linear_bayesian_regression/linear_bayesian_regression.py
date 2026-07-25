import numpy as np


class LinearBayesianRegression():
    
    def __init__(self, n, alpha, beta):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.mu = np.array([[0] for _ in range(n)])
        self.S = np.identity(n) / self.alpha
        return
    
    def learn(self, X, y):
        lamb = np.linalg.inv(self.S)
        S = np.linalg.inv(lamb + self.beta * np.dot(np.array(X).T, np.array(X)))
        # print(f"self.S:{self.S}")
        # print(np.array(X).T)
        # print(y)
        # print(np.dot(S, np.dot(np.array(X).T, y)))
        # print(np.dot(self.S, self.mu))
        # print(self.mu)
        mu = np.dot(S, self.beta * np.dot(np.array(X).T, y) + np.dot(lamb, self.mu))
        # print(mu)
        self.S = S
        self.mu = mu
        return
    
    def w_mostlike(self):
        return self.mu.T[0]
    
    def w_sample(self, n=1):
        # print(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]]))
        # print(self.mu.T)
        # print(self.S)
        return [np.random.multivariate_normal(self.mu.T[0], self.S).T for _ in range(n)]
    
    def w_distribute(self):
        return self.mu.T[0], self.S
    
    def y_mostlike(self, X):
        return np.dot(X, self.mu)
    
    def y_sample(self, X, n):
        return np.dot(X, self.w_sample(n))
    
    def y_distribute(self, X):
        return np.dot(X, self.mu).T[0], np.diag((1 / self.beta) + np.dot(X, np.dot(self.S, np.array(X).T)))
    
    
    
    