import numpy as np

def data_function(x):
    return 4*x**3 + x**2 - 17*x + 48

def poly_index(maxpoly):
    return(sum(np.arange(maxpoly + 1) + 1))

def sub_to_poly(X_train, X_test, maxpoly, remove_intercept = True, interaction = False):
    if interaction:
        X_train_npoly = X_train[:,int(remove_intercept):poly_index(maxpoly)]
        X_test_npoly = X_test[:,int(remove_intercept):poly_index(maxpoly)]
    else:
        X_train_npoly = X_train[:,int(remove_intercept):maxpoly+1]
        X_test_npoly = X_test[:,int(remove_intercept):maxpoly+1]

    return X_train_npoly, X_test_npoly
