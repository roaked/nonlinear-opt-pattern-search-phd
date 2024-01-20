import numpy as np
from scipy.optimize import line_search, golden
import matplotlib.pyplot as plt
import copy

#def hooke_jeeves(f, x_init, sigma_init, epsilon, alpha):
def hooke_jeeves_old(f, initial_point, initial_step_size, epsilon, acceleration_factor):
    x_k = initial_point
    A = initial_step_size
    k = j = 1

    while A > epsilon:
        j = 1

        while j <= 2:

            direction = x_k + acceleration_factor * (x_k - initial_point) - x_k
            trial_point, v_lambda = golden_section_search(f, x_k, direction, 0.0, A, epsilon)

            if f(trial_point) < f(x_k):
                # Trial is a success
                x_k = trial_point

            j += 1

        #Update x using acceleration
        x_k_plus_1 = x_k + acceleration_factor * (x_k - initial_point)
        x_k = x_k_plus_1
        k += 1
        j = 1

        # update step size
        A /= 2

    return x_k

def hooke_jeeves(f, initial_guess, step, epsilon, max_iter):
    """
    Hooke-Jeeves direct search method for unconstrained optimization.
    
    Parameters:
    - f: function to be minimized
    - initial_guess: initial guess
    - step: initial step size
    - epsilon: convergence tolerance
    - max_iter: maximum number of iterations
    
    Returns:
    - xmin: optimal solution
    - fmin: minimum value of the objective function
    """
    x = np.array(initial_guess, dtype=float)
    x_new = np.copy(x)
    fmin = f(x)
    n = len(initial_guess)

    for iter in range(1, max_iter + 1):
        x_old = np.copy(x)

        # Exploratory move
        for i in range(n):
            x_new[i] = x[i] + step
            if f(x_new) < fmin:
                x[i] = x_new[i]
                fmin = f(x_new)
            else:
                x_new[i] = x[i] - step
                if f(x_new) < fmin:
                    x[i] = x_new[i]
                    fmin = f(x_new)
                else:
                    x_new[i] = x[i]

        # Pattern move
        x_new = x + (x - x_old)
        if f(x_new) < fmin:
            x = x_new
            fmin = f(x_new)

        # Check for convergence
        if np.max(np.abs(x - x_old)) < epsilon:
            break

        # Reduce step size
        if f(x_new) >= f(x_old):
            step /= 2.0

    return x, fmin

class Function:
    def __init__(self, expression):
        self.expression = expression
        self.iterations = 0

    def reset_iterations(self):
        self.iterations = 0

    def calc(self, *args):
        """
        Calculates the value of the function for the given point.
        Function also track the number of calls.
        :param args: point
        :return: value of the function at the provided point
        """
        self.iterations += 1
        return self.expression(*args)

def hooke_jeeves_2(f, initial_guess, step, epsilon):
    x0 = copy.deepcopy(initial_guess)
    xb = copy.deepcopy(initial_guess)
    xp = copy.deepcopy(initial_guess)
    xn = None

    f.reset_iterations
    while True:
        if step <= epsilon:
            break
        xn = search(xp, f, step)
        f_xn = f.calc(*xn)
        f_xb = f.calc(*xb)
        if f_xn < f_xb:
            for i in range(0, len(xn)):
                xp[i] = 2*xn[i] - xb[i]
                xb[i] = xn[i]
        else:
            step /= 2
            xp = copy.deepcopy(xb)

    return xb


def search(xp, func, step):
    x = copy.deepcopy(xp)
    for i in range(0, len(xp)):
        p = func.calc(*x)
        x[i] += step
        n = func.calc(*x)
        if n > p:
            x[i] -= 2 * step
            n = func.calc(*x)
            if n > p:
                x[i] += step
    return x


def golden_section_search(f, initial_guess, direction, a, b, epsilon):
    # Golden section search to find the minimum along the direction
    v_lambda = golden(lambda lam: f(initial_guess + lam * direction), brack=(a, b), tol=epsilon)
    return initial_guess + v_lambda * direction, v_lambda

def func(x1, x2):
    return  (x1 - 2*x2)**2 + (x1 - 2)**4

f = Function(func)

a, b, epsilon, alpha = -1, 11, 0.02, 0.05
step, max_iter = 5, 50
init_guess = Xj = np.array([0.0, 3.0])
#trajectory = hooke_jeeves(f, Xj, initial_step_size, epsilon, alpha)
trajectory = hooke_jeeves_2(f, init_guess, step, epsilon)

print("Optimal solution:", trajectory)
