import numpy as np

def data_function(x):
    return 4*x**3 + x**2 - 17*x + 48

def poly_index(maxpoly):
    return(sum(np.arange(maxpoly + 1) + 1))