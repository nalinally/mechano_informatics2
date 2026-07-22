import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(x, s):
    return x + np.random.normal(loc=0, scale=s, size=len(x))

def linear(a, b, x):
    return a * x + b