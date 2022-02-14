import numpy as np
import math
from scipy.optimize import minimize
from typing import List, Tuple


def f_d2(x: List[float]) -> float:
    x = np.array(x)
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def f_d3(x: List[float]) -> float:
    x = np.array(x)
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2 + x[2]**2

def optimizer_dummy(f, N_x: int, bounds: List[Tuple[float]], N: int = 100) -> (float, List[float]):
    '''
    Optimizer aims to optimize a black-box function 'f' using the dimensionality
    'N_x', and box-'bounds' on the decision vector
    Input:
    f: function: taking as input a list of size N_x and outputing a float
    N_x: int: number of dimensions
    N: int: optional: Evaluation budget
    bounds: List of size N where each element i is a tuple conisting of 2 floats
            (lower, upper) serving as box-bounds on the ith element of x
    Return:
    tuple: 1st element: lowest value found for f, f_min
           2nd element: list/array of size N_x giving the decision variables
                        associated with f_min
    '''
    if N_x != len(bounds):
        raise ValueError('Nbr of variables N_x does not match length of bounds')

    ### Your code here
    # x = [np.mean(bounds[i]) for i in range(N_x)]
    ND = math.ceil(20/N_x)
    BO_test = BO(f, N_x, bounds, ND=ND)
    x_opt, y_opt = BO_test.opt_loop()

    x = x_opt.flatten().tolist()
    ###
    return f(x), x

def kernel(x, y, l2=0.1, sigma_f=1):
    '''
    Exponential kernel function
    @Input
    x: Nx x Dx (each data of x is a column vector)
    y: Nx x Dy
    @Return:Nx x Ny
    '''
    # Nx, Dx = np.shape(x)
    # Nx, Dy = np.shape(y)
    square = np.sum(x ** 2, 0).reshape(-1, 1) + np.sum(y ** 2, 0).reshape(1, -1) - 2 * (x.T @ y)
    return sigma_f ** 2 * np.exp(-0.5 * (1 / l2) * square)


def posterior(X, X_test, y, l2=0.1, sigma_y=1e-3):
    '''
    Compute the expectation and covariance by using a practical algorithm
    @Input
    X: Nx x Dx (each x sample is a column vector)
    X_test: Nx x D_test
    y: 1 x Dx
    @Return
    expect
    cov
    '''
    Nx, Dx = np.shape(X)
    # Nx, Dy = np.shape(X_test)
    K = kernel(X, X, l2)
    L = np.linalg.cholesky(K + sigma_y ** 2 * np.eye(Dx))
    alpha_temp = np.linalg.solve(L, y.T)  # make y as a col vector
    alpha = np.linalg.solve(L.T, alpha_temp)
    K_s = kernel(X, X_test, l2)
    expect = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    K_ss = kernel(X_test, X_test, l2)
    cov = np.diag(K_ss) - v.T @ v

    return expect, cov


def define_lcb(X, y, l2=0.1, sigma_y=0, kappa=5):
    '''
    Define the lcb function for the minimization problem to determine the next sample position
    @Input
    X: Nx x Dx (each x sample is a column vector)
    y: 1 x Dx
    @Return
    lower_confidence_bound function
    '''
    def lower_confidence_bound(X_test):
        X_test = X_test.reshape(-1,1)
        expect, cov = posterior(X, X_test, y, l2, sigma_y)
        sigma = np.sqrt(cov)
        return expect.item() - kappa * sigma.item()
    return lower_confidence_bound

class BO:
    '''
    Minimal Bayesian Optimization class
    TODO: replace sampling with some low-discrepency methods e.g. sobol
    TODO: rescale
    TODO: tune hyperparameters
    TODO: Rewrite in Casadi
    '''
    def __init__(self, black_fn, Nx, bounds, ND=5):

        self.black_fn = black_fn
        self.Nx = Nx
        self.ND = ND
        self.bounds = bounds

        X_train, Y_train = self.collect_data()
        self.X_train = X_train
        self.Y_train = Y_train

    def collect_data(self):
        # mesh grid
        Nx = self.Nx
        ND = self.ND
        black_fn = self.black_fn
        range_list = [np.linspace(bound[0], bound[1], ND) for bound in bounds]
        grid_list = (np.array(np.meshgrid(*range_list)).T.reshape(-1, Nx)).tolist()
        output_list = []
        for grid in grid_list:
            output_list += [black_fn(grid)]

        return np.array(grid_list).T, np.array(np.array(output_list).reshape(1, -1))

    def opt_loop(self, N_iter=100):
        ## initialize with a random point
        x_k = np.array([np.random.uniform(bound[0], bound[1], 1) for bound in bounds])
        for i in range(N_iter):
            ## calculate f(x) for a given x
            print(x_k)
            x_k_list = x_k.flatten().tolist()
            y_k = np.array(self.black_fn(x_k_list)).reshape(-1,1)

            ## update training set
            self.X_train = np.hstack([self.X_train, x_k])
            self.Y_train = np.hstack([self.Y_train, y_k])

            ## acquisition fn for the next x
            lcb_fn = define_lcb(self.X_train, self.Y_train, l2=0.1, sigma_y=1e-3, kappa=2)
            x0_list = x_k_list
            res = minimize(lcb_fn, x0_list, method='SLSQP', bounds=self.bounds)
            x_k = (res.x).reshape(-1,1)

            ## update optimum
            if i == 0:
                y_opt = y_k
                x_opt = x_k
            else:
                if y_k.item() <= y_opt.item():
                    y_opt = y_k
                    x_opt = x_k

        return x_opt, y_opt

if __name__ == "__main__":
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    print(optimizer_dummy(f_d2, 2, bounds, 100))

    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    print(optimizer_dummy(f_d3, 3, bounds, 100))

    # bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    # BO_test = BO(f_d2, 2, bounds, ND=20)
    # x_opt, y_opt = BO_test.opt_loop()
    #
    # bounds = [(-2.0, 2.0), (-2.0, 2.0),  (-2.0, 2.0)]
    # BO_test = BO(f_d3, 3, bounds, ND=10)
    # x_opt, y_opt = BO_test.opt_loop()


