import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import multivariate_normal

import linear_bayesian_regression as lbr
sys.path.append("../")
import common.generate_data as gendata
import common.visualize as vis

alpha = 100
beta = 0.01
a = 3
b = -1
sigma = 3
x_range = [0, 10]
y_range = [-10, 40]

def main():
    lbr_regression = lbr.LinearBayesianRegression(2, alpha, beta)
    visualizer = vis.Visualize()
    
    mses_list = []
    
    for _ in range(1):
    
        lbr_regression.__init__(2, alpha, beta)
    
        mses = []
        
        rng = np.random.default_rng()
        x = rng.random(100) * (x_range[1] - x_range[0]) + x_range[0]
        def y_func(x):
            return gendata.linear(a, b, x)
        def y_noise_func(x):
            return gendata.add_gaussian_noise(x, sigma)
        y = [y_noise_func(y_func(x_)) for x_ in x]
            
        mu, S = lbr_regression.w_distribute()
        # print(mu)
        # print(S)
        rv = multivariate_normal(mu, S)
        def f(x, y):
            return rv.pdf([x, y])
        visualizer.reset()
        visualizer.set_figsize(18, 6)
        visualizer.draw_2_variable_func_colormap(f, [-5, 5], [-5, 5], f"n=0", "w0", "w1")
        w = lbr_regression.w_sample(20)
        x_lin = np.linspace(x_range[0], x_range[1], 200)
        x_data_for_ydis = np.hstack([[[1] for _ in range(200)], np.array([x_lin]).T])
        y_distribute = lbr_regression.y_distribute(x_data_for_ydis)
        y_lin = np.linspace(-10, 40, 200)
        z = []
        for ydis in np.array(y_distribute).T:
            rv = multivariate_normal(ydis[0], ydis[1])
            z.append(rv.pdf(y_lin))
        X, Y = np.meshgrid(x_lin, y_lin)
        visualizer.draw_1_variable_func(y_func, x_range, False, "x vs y", f"y = {a}x + {b}", "x", "y", "blue")
        for w_ in w:
            def g(x):
                return gendata.linear(w_[1], w_[0], x)
            visualizer.draw_1_variable_func(g, x_range, True, "", f"y = {w_[1]}x + {w_[0]}", "x", "y", "red")
        visualizer.draw_2_variable_data_colormap(X, Y, np.array(z).T, f"pdf of (x, y): n={0}", "x", "y")
        # visualizer.show()
        visualizer.save(f"../../figs/linear_bayesian_regression/a100_b001_sigma3/0.png")
        # mses.append(linear_func_mse(a, b, mu[1], mu[0], x_range[0], x_range[1]))
        
        for i in range(len(x))[1:]:
            x_data = x[:i]
            y_data = y[:i]
            input = np.hstack([[[1] for _ in range(len(x_data))], np.array([x_data]).T])
            lbr_regression.learn(input, np.array(y_data))
            mu, S = lbr_regression.w_distribute()
            w = lbr_regression.w_sample(20)
            x_lin = np.linspace(x_range[0], x_range[1], 200)
            x_data_for_ydis = np.hstack([[[1] for _ in range(200)], np.array([x_lin]).T])
            y_distribute = lbr_regression.y_distribute(x_data_for_ydis)
            y_lin = np.linspace(-10, 40, 200)
            z = []
            for ydis in np.array(y_distribute).T:
                rv = multivariate_normal(ydis[0], ydis[1])
                z.append(rv.pdf(y_lin))
            X, Y = np.meshgrid(x_lin, y_lin)
            # print(mu)
            # print(S)
            rv = multivariate_normal(mu, S)
            def f(x, y):
                return rv.pdf([x, y])
            visualizer.reset()
            visualizer.set_figsize(18, 6)
            visualizer.draw_2_variable_func_colormap(f, [-5, 5], [-5, 5], f"pdf of (w0, w1): n={i}", "w0", "w1")
            visualizer.draw_1_variable_func(y_func, x_range, False, "x vs y", f"y = {a}x + {b}", "x", "y", "blue")
            visualizer.draw_1_variable_data(x_data, y_data, True, True, "", f"train data", "x", "y")
            for w_ in w:
                def g(x):
                    return gendata.linear(w_[1], w_[0], x)
                visualizer.draw_1_variable_func(g, x_range, True, "", f"y = {w_[1]}x + {w_[0]}", "x", "y", "red")
            visualizer.draw_2_variable_data_colormap(X, Y, np.array(z).T, f"pdf of (x, y): n={i}", "x", "y")
            # visualizer.show()
            visualizer.save(f"../../figs/linear_bayesian_regression/a100_b001_sigma3/{i}.png")
            mses.append(linear_func_mse(a, b, mu[1], mu[0], x_range[0], x_range[1]))
        
        mses_list.append(mses)
    
    plt.figure(figsize=(15, 8))
    for i, mses in zip(range(len(mses_list)), mses_list):
        plt.subplot(2, 3, i+1)
        plt.plot([np.log10(i+1) for i in range(len(mses))], np.log10(mses))
        plt.title(f"{i}")
        plt.xlabel("log(iteration)")
        plt.ylabel("log(mse) true and predicted func")
        plt.ylim(-2, 2)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../../figs/linear_bayesian_regression/mse_alpha{alpha}_beta{beta}.png", format="png")
    
def linear_func_mse(a1, b1, a2, b2, x1, x2):
    return ((a1 - a2)**2 * (x2**3 - x1**3) / 3 +
            (a1 - a2) * (b1 - b2) * (x2**2 - x1**2) +
            (b1 - b2)**2 * (x2 - x1)) / (x2 - x1)

if __name__=="__main__":
    main()