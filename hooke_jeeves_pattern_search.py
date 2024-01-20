import numpy as np
from scipy.optimize import line_search, golden
import matplotlib.pyplot as plt
import copy

def calc(func, *args): 
    return func(*args) # value of function at this point

def hooke_jeeves(f, initial_guess, step, epsilon):
    xb = copy.deepcopy(initial_guess)
    xp = copy.deepcopy(initial_guess)
    xn = None
    trajectory = [np.array(xp)]  # initial guess
    while True:
        if step <= epsilon:
            break

        xn = search(xp, func, step)
        f_xn = func(*xn)
        f_xb = func(*xb)

        if f_xn < f_xb:
            # golden section search for line minimization
            direction = np.array(xn) - np.array(xp)
            a, b = 0, step
            new_xb, _ = golden_section_search(func, xp, direction, a, b, epsilon)
            xb = new_xb
            xp = 2 * np.array(xn) - np.array(xb)
        else:
            step /= 2
            xp = copy.deepcopy(xb)

        trajectory.append(np.array(xp))
    return xb, np.array(trajectory)


def search(xp, func, step):
    x = copy.deepcopy(xp)
    for i in range(0, len(xp)):
        p = func(*x)
        x[i] += step
        n = func(*x)
        if n > p:
            x[i] -= 2 * step
            n = func(*x)
            if n > p:
                x[i] += step
    return x

def golden_section_search(func, initial_guess, direction, a, b, epsilon):
    # Golden section search to find the minimum along the direction
    v_lambda = golden(lambda lam: func(*(initial_guess + lam * np.array(direction))), brack=(a, b), tol=epsilon)
    return initial_guess + v_lambda * np.array(direction), v_lambda

def func(x1, x2):
    return  (x1 - 2*x2)**2 + (x1 - 2)**4

#f = Function(func)

a, b, epsilon, alpha = -5, 10, 0.1, 0.05
step, max_iter = 5, 50
init_guess = Xj = np.array([0.0, 3.0])
#trajectory = hooke_jeeves(f, Xj, initial_step_size, epsilon, alpha)
xb, trajectory = hooke_jeeves(func, init_guess, step, epsilon)

print("Optimal solution:", trajectory)
print("Optimal solution:", xb)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', label='x1 vs x2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()