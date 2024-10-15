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



class AdaGrad:

class RMSProp:

def grad_OLS():
    def grad_fun(X, y, theta):
        n = y.shape[0]
        return -(2.0/n) * X.T @ (y - X @ theta)
    return grad_fun

def grad_ridge(lmb):
    def grad_fun(X, y, theta):
        n = y.shape[0]
        return -(2.0/n) * X.T @ (y - X @ theta) + 2*lmb*theta
    return(grad_fun)


class GradientDescent:
    def __init__(
        self, 
        gradient,
        n_iter = 1000,

        **gradient_args
        ):
        self.gradient = gradient
        self.gradient_args = gradient_args
        self.theta = None
