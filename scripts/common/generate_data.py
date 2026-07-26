import numpy as np
from collections.abc import Iterable

def add_gaussian_noise(x, s):
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return x + np.random.normal(loc=0, scale=s, size=len(x))
    else:
        return x + np.random.normal(loc=0, scale=s, size=1)

def linear(a, b, x):
    return a * x + b
