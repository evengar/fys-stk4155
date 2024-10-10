import numpy as np

def gradient_OLS(X, y, theta):
    n = y.shape[0]
    return -(2.0/n) * X.T @ (y - X @ theta)

def gradient_ridge(X, y, theta, lmb):
    n = y.shape[0]
    return 2.0/n * X.T @ (X @ theta-y) + 2*lmb*theta

def AdaGrad(update_term, gradient, Giter, delta = 1e-8):
    Giter += gradient*gradient
    update = update_term / (delta + np.sqrt(Giter))
    return Giter, update

def RMSProp(update_term, gradient, Giter, rho, delta=1e-8):
    Giter = rho*Giter + (1-rho)*gradient*gradient
    update = update_term / (delta+np.sqrt(Giter))
    return Giter, update

def ADAM(gradient, first_moment, second_moment, beta1, beta2, itr, delta=1e-8):
    first_moment = beta1*first_moment + (1-beta1)*gradient
    second_moment = beta2*second_moment + (1-beta2)*gradient*gradient

    first_term = first_moment/(1.0-beta1**itr)
    second_term = second_moment/(1.0-beta2**itr)
    update = eta*first_term/(np.sqrt(second_term)+delta)

    return first_moment, second_moment, update

def GD_inner(eta, theta, moments, gradient, momentum=False, gamma=None, adaptive_fun=None, adam = False, **kwargs):
    adaptive = adaptive_fun is not None

    if adam:
        first_moment, second_moment = moments
        first_moment, second_moment, update = ADAM(gradient, first_moment, second_moment, **kwargs)
        theta -= update
        return theta, first_moment, second_moment
    else:
        Giter, change = moments
        update = eta * gradient
        if momentum:
            update += gamma * change
            change = update
        if adaptive:
            Giter, update = adaptive_fun(update, gradient, Giter, **kwargs)
    
    theta -= update
    return theta, Giter, change

def GD(X, y, eta, n_iter, gradient_fun=gradient_OLS, 
       momentum=False, gamma=None, adaptive_fun=None, 
       adam=False, gradient_args={}, **kwargs):
    theta = np.random.randn(X.shape[1], 1)
    
    # moment 1 and 2 of ADAM
    # Giter and change if not ADAM
    moments = [0, 0]

    for i in range(n_iter):
        gradient = gradient_fun(X, y, theta, **gradient_args)
        if adam:
            theta, moments[0], moments[1] = GD_inner(eta, theta, moments, gradient, momentum, gamma, adaptive_fun, adam, itr = i+1, **kwargs)
        else:
            theta, moments[0], moments[1] = GD_inner(eta, theta, moments, gradient, momentum, gamma, adaptive_fun, adam, **kwargs)
    return theta

def SGD(X, y, eta, M, n_epochs, gradient_fun=gradient_OLS, momentum=False, gamma=None, adaptive_fun=None, adam=False, gradient_args={}, **kwargs):
    n = y.shape[0]
    m = int(n/M)
    xy = np.column_stack([X,y]) # for shuffling x and y together
    theta = np.random.randn(X.shape[1], 1)
    # moment 1 and 2 of ADAM
    # Giter and change if not ADAM
    moments = [0, 0]

    for i in range(n_epochs):
        Giter = 0.0
        np.random.shuffle(xy)
        for j in range(m):
            random_index = M * np.random.randint(m)
            xi = xy[random_index:random_index+5, :-1]
            yi = xy[random_index:random_index+5, -1:]
            gradient = (1/M)*gradient_fun(xi, yi, theta, **gradient_args)
            if adam:
                theta, moments[0], moments[1] = GD_inner(eta, theta, moments, gradient, momentum, gamma, adaptive_fun, adam,  itr = i+1, **kwargs)
            else:
                theta, moments[0], moments[1] = GD_inner(eta, theta, moments, gradient, momentum, gamma, adaptive_fun, adam, **kwargs)
    return theta
