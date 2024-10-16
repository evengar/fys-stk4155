import numpy as np

class ADAM:
    def __init__(self, beta1=0.9, beta2=0.999, delta=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        self.first_moment = 0.0
        self.second_moment = 0.0

    def reset(self):
        self.first_moment = 0.0
        self.second_moment = 0.0

    def calculate(self, learning_rate, gradient, current_iter):
        self.first_moment = self.beta1*self.first_moment + (1-self.beta1)*gradient
        self.second_moment = self.beta2*self.second_moment + (1-self.beta2)*gradient*gradient

        first_term = self.first_moment/(1.0-self.beta1**current_iter)
        second_term = self.second_moment/(1.0-self.beta2**current_iter)
        update = learning_rate*first_term/(np.sqrt(second_term)+self.delta)

        return update



#class AdaGrad:

#class RMSProp:

def grad_OLS():
    def grad_fun(X, y, theta):
        n = y.shape[0]
        return (2.0/n) * X.T @ (X @ theta - y)
    return grad_fun

def grad_ridge(lmb):
    def grad_fun(X, y, theta):
        n = y.shape[0]
        return -(2.0/n) * X.T @ (X @ theta - y) + 2*lmb*theta
    return(grad_fun)


class GradientDescent:
    def __init__(self, learning_rate, gradient, momentum = False, adaptive = None, n_iter = 1000):
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.momentum = momentum
        self.adaptive = adaptive
        self.n_iter = n_iter
        self.theta = None
        self.n = None
        if self.momentum:
            self.momentum_change = 0.0
    def _initialize_vars(self, X):
        self.theta = np.random.randn(X.shape[1], 1)
        self.n = X.shape[0]

    def _gd(self, grad, X, y, current_iter):
        if self.adaptive is None:
            update = self.learning_rate * grad
            if self.momentum:
                update += gamma * self.momentum_change
                self.momentum_change = update
        else:
            update = self.adaptive.calculate(self.learning_rate, grad, current_iter)

        return update

    def descend(self, X, y):
        self._initialize_vars(X)
        for i in range(self.n_iter):
            grad = self.gradient(X, y, self.theta)
            update = self._gd(grad, X, y, i+1)
            self.theta -= update


