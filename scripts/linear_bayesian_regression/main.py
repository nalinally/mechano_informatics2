import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import linear_bayesian_regression as lbr
sys.path.append("../")
import common.generate_data as gendata

alpha = 10
beta = 10
a = 3
b = -1
sigma = 0.1

def main():
    lbr_regression = lbr.LinearBayesianRegression(2, alpha, beta)
    
    rng = np.random.default_rng()
    x = rng.random(100)
    y = gendata.add_gaussian_noise(gendata.linear(a, b, x), sigma)
    
    print(x)
    
    for i in range(len(x)):
        input = np.hstack([[[1] for _ in range(i)], np.array(x[:i]).T])
        lbr_regression.learn(input, y[:i])
        print(lbr_regression.w_sample())

if __name__=="__main__":
    main()