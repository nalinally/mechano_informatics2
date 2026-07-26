import numpy as np

class GaussianProcessRegression():
    
    def __init__(self, k):
        self.k = k
        self.K = None
        self.K_inv = None
        self.X = None
        self.y = None

    @classmethod
    def optimize_hiper_param(self, X, y, n, k, dkdtheta, theta0, th=0.01, iter=1000, alpha=0.1, epsilon=1e-5):
        def L(theta):
            K = [[k(theta, x, x_) for x in X] for x_ in X] + np.identity(len(X)) * epsilon
            K_inv = np.linalg.inv(K)
            return -np.log(np.linalg.det(K)) - np.dot(np.array(y).T, np.dot(K_inv, y))
        def dLdtheta(theta):
            K = [[k(theta, x, x_) for x in X] for x_ in X] + np.identity(len(X)) * epsilon
            K_inv = np.linalg.inv(K)
            K_inv_y = np.dot(K_inv, y)
            dKdtheta = np.array([[dkdtheta(theta, x, x_) for x in X] for x_ in X])
            return [-np.trace(np.dot(K_inv, dKdtheta[:, :, i])) + np.dot(K_inv_y.T, np.dot(dKdtheta[:, :, i], K_inv_y)) for i in range(len(dKdtheta[0][0]))]
        theta = theta0
        for _ in range(iter):
            diff = dLdtheta(theta)
            diff_norm = np.linalg.norm(diff, ord=2)
            print(f"{L(theta)}, {theta}, {diff}, {diff_norm}")
            if np.isnan(L(theta)):
                K = [[k(theta, x, x_) for x in X] for x_ in X] + np.identity(len(X)) * epsilon
                K_inv = np.linalg.inv(K)
                print(K)
                print(K_inv)
                return
            if diff_norm <= th:
                print("[GPR.opthiperparam] converged.")
                return theta
            else:
                theta -= diff / diff_norm * alpha
        print("[GPR.opthiperparam] max iter reached.")
        return theta

    def learn(self, X, y):
        self.K = [[self.k(x, x_) for x_ in X] for x in X]
        self.K_inv = np.linalg.inv(self.K)
        self.X = X
        self.y = y
        
    def y_dist(self, X):
        kN = np.array([[self.k(x, x_) for x_ in self.X] for x in X])
        K_input = [[self.k(x, x_) for x_ in X] for x in X]
        mu = np.dot(kN, np.dot(self.K_inv, self.y))
        S = K_input - np.dot(kN, np.dot(self.K_inv, kN.T))
        return mu, S
    
    def y(self, X):
        mu, _ = self.y_dist(X)
        return mu
                

        
        