import numpy as np
import sys

sys.path.append("../")
import common.visualize as vis

class GaussianProcessRegression():
    
    def __init__(self, k):
        self.k = lambda x, x_: (k(x, x_) + k(x_, x)) / 2
        self.K = None
        # self.K_inv = None
        self.X = None
        self.y = None

    @classmethod
    def optimize_hiper_param(self, X, y, n, k, dkdtheta, theta0, visualizer, th=0.01, iter=1000, alpha=2):
        def L(theta):
            K = np.array([[k(theta, x, x_) for x in X] for x_ in X])
            K += np.eye(len(K)) * 1e-10
            x = np.linalg.solve(K, y)
            return -np.log(np.linalg.det(K)) - np.array(y).T @ x
        def dLdtheta(theta):
            K = np.array([[k(theta, x, x_) for x in X] for x_ in X])
            K += np.eye(len(K)) * 1e-10
            x = np.linalg.solve(K, y)
            dKdtheta = np.array([[dkdtheta(theta, x, x_) for x in X] for x_ in X])
            return [-np.trace(np.linalg.solve(K, dKdtheta[:, :, i])) + x.T @ dKdtheta[:, :, i] @ x for i in range(len(dKdtheta[0][0]))]
        x_vis = np.linspace(-3, 3, 200)
        y_vis = np.linspace(-3, 3, 200)
        X_, Y_ = np.meshgrid(x_vis, y_vis)
        Z = [[L([x_, y_]) + y_*0 for x_ in x_vis] for y_ in y_vis]
        Z = -np.log(np.max(Z) - Z + 1e-3)
        visualizer.draw_2_variable_data_colormap(X_, Y_, Z, "log(L}+elog(length_scale)", "log(sigma)", "log(length_scale)")
        # theta = theta0
        # for _ in range(iter):
        #     diff = dLdtheta(theta)
        #     diff_norm = np.linalg.norm(diff, ord=2)
        #     # print(f"{L(theta)}, {theta}, {diff}, {diff_norm}")
        #     # if np.isnan(L(theta)):
        #     #     K = [[k(theta, x, x_) for x in X] for x_ in X]
        #     #     K_inv = np.linalg.solve(K, np.eye(len(K)))
        #     #     print(K)
        #     #     print(K_inv)
        #     #     return
        #     if diff_norm <= th:
        #         print(f"[GPR.opthiperparam] converged. dLdtheta:{diff_norm} theta:{theta}")
        #         return theta
        #     else:
        #         theta_list = [theta + (diff / diff_norm * a) for a in np.linspace(0, diff_norm*alpha, 100)]
        #         # dLdtheta_list = [np.abs(np.dot(dLdtheta(theta_), diff) / diff_norm) for theta_ in theta_list]
        #         L_list = [L(theta_) for theta_ in theta_list]
        #         theta = theta_list[np.argmax(L_list)]
        #         # theta += diff / diff_norm * alpha
        # print(f"[GPR.opthiperparam] max iter reached. dLdtheta:{diff_norm} theta:{theta}")
        row, col = np.unravel_index(np.argmax(Z), Z.shape)
        theta = [X_[row][col], Y_[row][col]]
        visualizer.draw_1_variable_data([theta[0]], [theta[1]], True, True, "", "optimized point", color="red")
        print(f"[GPR.opthiperparam] theta:{theta} L:{L(theta)}")
        return theta

    def learn(self, X, y):
        self.K = np.array([[self.k(x, x_) for x_ in X] for x in X])
        self.K += np.eye(len(self.K)) * 1e-10
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
                

        
        