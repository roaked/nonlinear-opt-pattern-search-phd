import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

def minimize_augmented_lagrangian(x0, rho, lambda_, max_iter=100):
    for _ in range(max_iter):
        # Minimize the augmented Lagrangian with respect to primal variables
        x_optimal = minimize(
            fun=lambda x: augmented_lagrangian(x, rho, lambda_),
            x0=x0,
            constraints=[
                {'type': 'ineq', 'fun': inequality_constraint},
                {'type': 'eq', 'fun': equality_constraint}
            ],
            bounds=[(0, 5), (0, 5)]
        ).x

        lambda_ = np.maximum(0, lambda_ + rho * equality_constraint(x_optimal))

        rho *= 2 # penalty

        x0 = x_optimal

    return x_optimal


start_point = np.array([0, 3])
rho = 1.0
lambda_ = 0.0

optimal_solution = minimize_augmented_lagrangian(start_point, rho, lambda_)
print("Optimal Solution:", optimal_solution)
print("Objective Value:", objective_function(optimal_solution))


"""
min (x1^4 - 2x1^2 x2 + x1^2 + x1 x2^2 - 2x1 + 4)
subject to:

0.25x1^2 + 0.75x2^2 <= 1
2x1^2  + x2^2 = 2
x1,x2 belongs to 0 to 5 
"""