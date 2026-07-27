import numpy as np

class GaussianProcessRegression():
    
    def __init__(self, k):
        self.k = lambda x, x_: (k(x, x_) + k(x_, x)) / 2
        self.K = None
        # self.K_inv = None
        self.X = None
        self.y = None

    @classmethod
    def optimize_hiper_param(self, X, y, n, k, dkdtheta, theta0, th=0.01, iter=1000, alpha=0.1):
        def L(theta):
            K = [[k(theta, x, x_) for x in X] for x_ in X]
            x = np.linalg.solve(K, y)
            return -np.log(np.linalg.det(K)) - np.array(y).T @ x
        def dLdtheta(theta):
            K = [[k(theta, x, x_) for x in X] for x_ in X]
            x = np.linalg.solve(K, y)
            dKdtheta = np.array([[dkdtheta(theta, x, x_) for x in X] for x_ in X])
            return [-np.trace(np.linalg.solve(K, dKdtheta[:, :, i])) + x.T @ dKdtheta[:, :, i] @ x for i in range(len(dKdtheta[0][0]))]
        theta = theta0
        for _ in range(iter):
            diff = dLdtheta(theta)
            diff_norm = np.linalg.norm(diff, ord=2)
            # print(f"{L(theta)}, {theta}, {diff}, {diff_norm}")
            # if np.isnan(L(theta)):
            #     K = [[k(theta, x, x_) for x in X] for x_ in X]
            #     K_inv = np.linalg.solve(K, np.eye(len(K)))
            #     print(K)
            #     print(K_inv)
            #     return
            if diff_norm <= th:
                print(f"[GPR.opthiperparam] converged. theta:{theta}")
                return theta
            else:
                theta += diff / diff_norm * alpha
        print(f"[GPR.opthiperparam] max iter reached. theta:{theta}")
        return theta

    def learn(self, X, y):
        self.K = [[self.k(x, x_) for x_ in X] for x in X]
        # self.K_inv = np.linalg.inv(self.K)
        self.X = X
        self.y = y
        
    def y_dist(self, X):
        kN = np.array([[self.k(x, x_) for x_ in self.X] for x in X])
        K_input = [[self.k(x, x_) for x_ in X] for x in X]
        x = np.linalg.solve(self.K, self.y)
        mu = kN @ x
        x = np.array([np.linalg.solve(self.K, kN_.T) for kN_ in kN]).T
        S = K_input - kN @ x
        S = (S + S.T) / 2
        corr_S = -np.linalg.eigvalsh(S).min()
        S += np.eye(len(S)) * corr_S
        print(f"[GPR.y_dist] corr of S {corr_S}, min of eigval {np.linalg.eigvalsh(S).min()}")
        return mu, S
    
    def y(self, X):
        mu, _ = self.y_dist(X)
        return mu
                

        
        