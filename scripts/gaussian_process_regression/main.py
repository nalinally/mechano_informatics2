
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys

import gaussian_process_regression as gpr
sys.path.append("../")
import common.generate_data as gendata
import common.visualize as vis
import kernel_func as kernel

x_range = [0, 10]
sigma = 0

def main():
    x = np.random.random(50) * (x_range[1] - x_range[0]) + x_range[0]
    f = lambda x: np.sin(x) + np.cos(1.5*x)
    y = gendata.add_gaussian_noise([f(x_) for x_ in x], sigma)

    visualizer = vis.Visualize()

    for i in range(len(x)):
        x_input = x[:i+1]
        y_train = y[:i+1]
        y_mean = np.mean(y_train)
        y_input = y_train - y_mean

        theta = gpr.GaussianProcessRegression.optimize_hiper_param(
                np.array([x_input]).T, np.array(y_input).T, 2, 
                lambda theta, x1, x2: kernel.RBF(theta[0], theta[1])(x1, x2), 
                lambda theta, x1, x2: kernel.RBF_diff(theta[0], theta[1])(x1, x2),
                [1, 1])
        gpr_regression = gpr.GaussianProcessRegression(kernel.RBF(theta[0], theta[1]))

        gpr_regression.learn(np.array([x_input]).T, np.array(y_input).T)
        visualizer.reset()
        visualizer.set_figsize(8, 6)
        visualize_data(visualizer, gpr_regression, i, y_mean, x_input, y_train, f)
        # visualize_y_dist(visualizer, gpr_regression, i, y_mean)
        visualizer.show()

def visualize_data(visualizer, gpr, i, y_mean, x_sample, y_sample, f):
    x = np.linspace(x_range[0], x_range[1], 300)
    n_sample = 5
    mu, S = gpr.y_dist(np.array([x]).T)
    s = [np.sqrt(s2) for s2 in np.diag(S)]
    visualizer.draw_1_variable_func(f, x_range, False, f"x vs y: n={i}", f"true func", "x", "y")
    visualizer.draw_1_variable_data(x, mu + y_mean, True, False, "", f"predicted", "", "", "red")
    visualizer.fill_1_variable_data(x, mu + y_mean + s, mu + y_mean - s, "lime")
    visualizer.draw_1_variable_data(x_sample, y_sample, True, True, "", f"train data", "x", "y")
    # for j in range(n_sample):
    #     visualizer.draw_1_variable_data(x, np.random.multivariate_normal(mu, S) + y_mean, True, False, "", f"sample{j}", "", "", "green")

def visualize_y_dist(visualizer, gpr, i, y_mean):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = []
    mu, S = gpr.y_dist(np.array([x]).T)
    for mu_, s_ in zip(mu, np.diag(S)):
        rv = multivariate_normal(mu_, s_)
        Z.append(rv.pdf(y))
    visualizer.draw_2_variable_data_colormap(X, Y, np.array(Z).T, f"pdf of (x, y): n={i}", "x", "y")    



if __name__=="__main__":
    main()