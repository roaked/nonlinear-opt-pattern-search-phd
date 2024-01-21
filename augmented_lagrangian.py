import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from scipy.optimize import golden
import scipy.linalg as la

"""
min (x1^4 - 2x1^2 x2 + x1^2 + x1 x2^2 - 2x1 + 4)
subject to:

0.25x1^2 + 0.75x2^2 <= 1
2x1^2  + x2^2 = 2
x1,x2 belongs to 0 to 5 
"""

def objective_function(x):
    return x[0]**4 - 2*x[0]**2*x[1] + x[0]**2 + x[0]*x[1]**2 - 2*x[0] + 4

def inequality_constraint(x):
    return 0.25*x[0]**2 + 0.75*x[1]**2 - 1

def equality_constraint(x):
    return 2*x[0]**2 + x[1]**2 - 2

def augmented_lagrangian(x, rho, lambda_):
    return objective_function(x) + rho/2 * max(0, inequality_constraint(x))**2 + lambda_ * equality_constraint(x)

def augmented_lagrangian_derivative(x, rho, lambda_):
    df_dx1 = 4*x[0]**3 - 4*x[0]*x[1] + 2*x[0] - 2 + rho * x[0] * inequality_constraint(x) + 2 * lambda_ * x[0] * equality_constraint(x)
    df_dx2 = -2*x[0]**2 + 2*x[1]*x[0] + 2 * rho * x[1] * inequality_constraint(x) + 2 * lambda_ * x[1] * equality_constraint(x)
    return np.array([df_dx1, df_dx2])

def plot_solution_over_time(solutions):
    solutions = np.array(solutions)
    plt.figure(figsize=(8, 6))
    plt.plot(solutions[:, 0], solutions[:, 1], marker='o', linestyle='-', color='b')
    plt.scatter(solutions[-1, 0], solutions[-1, 1], color='r', label='Final Solution')
    plt.title('Solution Over Time')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

def minimize_augmented_lagrangian(x0, rho, lambda_, max_iter, epsilon, alpha1, alpha2):
    for _ in range(max_iter):
        # Calculate the augmented Lagrangian
        L = augmented_lagrangian(x0, rho, lambda_)
        g = augmented_lagrangian_derivative(x0, rho, lambda_)
        d = -g
        rho *= 1.1  # penalty
        print(L, g)

        iter_count = 0
        max_inner_iter = 30

        while alpha2 - alpha1 > epsilon and iter_count < max_inner_iter:
            print(L)
            alpha = (np.sqrt(5) + 1) / 2 * alpha1 - (np.sqrt(5) - 1) / 2 * alpha2
            L_new = augmented_lagrangian(x0 + alpha * d, rho, lambda_)
            if L_new <= L + alpha * 0.1 * g.dot(d):
                alpha2 = alpha
            else:
                alpha1 = alpha
            iter_count += 1

        x0 = x0 + alpha * d
        lambda_ = np.maximum(0, lambda_ + rho * equality_constraint(x0))

    return x0, lambda_


start_point = np.array([-10, 10])
rho = 1.0
lambda_ = 0.1
max_iter = 50
epsilon = 0.02
alpha1, alpha2 = 0.00001, 1

optimal_solution, lambda_ = minimize_augmented_lagrangian(start_point, rho, lambda_, max_iter, epsilon, alpha1, alpha2)
print("Optimal Solution [x1, x2] =", optimal_solution) # <1
print("Cost Value:", objective_function(optimal_solution)) # 22
print("Lagrangian Multipliers:", lambda_) # 2



"""
min (x1^4 - 2x1^2 x2 + x1^2 + x1 x2^2 - 2x1 + 4)
subject to:

0.25x1^2 + 0.75x2^2 <= 1
2x1^2  + x2^2 = 2
x1,x2 belongs to 0 to 5 
"""