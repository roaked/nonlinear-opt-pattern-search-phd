import numpy as np
import math
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

MAX_ITERATIONS = 100_000
EPSILON = 0.02

# def fletcher_reeves_method(objective_function, gradient, x1, MAX_ITERATIONS = MAX_ITERATIONS, EPSILON = EPSILON):
#     y1 = x1
#     grad_k = gradient(y1)
#     d = - grad_k  

#     for k in range(MAX_ITERATIONS):
#         varlambda = golden_section_line_search(objective_function, y1, d) 

#         y1 = y1 + varlambda * d 
#         grad_k1 = gradient(y1)  

#         alpha = np.dot(grad_k1, grad_k1) / np.dot(grad_k, grad_k) 

#         d = -grad_k1 + alpha * d  

#         if np.linalg.norm(grad_k1) < EPSILON:
#             break  

#         grad_k = grad_k1  # Update gradient for the next iteration

#     return y1, k+1

# def golden_section_line_search(objective_function, x):
#     a, b = -1.0, 3.0  # init interval for gold sec

#     def line_search_function(alpha):
#         return objective_function(x + alpha)

#     result = minimize_scalar(line_search_function, method='golden', bracket=(a, b))
#     return result.x

def golden_section_method(a_l, b_l, epsilon):
    a = 0.618  # Golden ratio

    while abs(b_l - a_l) > epsilon:
        A_l = a_l + (1 - a) * (b_l - a_l)
        P_l = a_l + a * (b_l - a_l)

        res_B_A_l = objective_function(A_l)
        res_B_P_l = objective_function(P_l)

        if res_B_A_l < res_B_P_l:
            b_l = P_l
        else:
            a_l = A_l

    optimal_x = (a_l + b_l) / 2
    optimal_value = objective_function(optimal_x)

    return optimal_x, optimal_value

def objective_function(x):
    return (x - 2)**4 + (x - 2 * x)**2

def objective_function2(x):
    x1 = x[0]
    return (6 * math.e) ** (-2*x1) + (2 * x1)** 2

def gradient(x):
    x1, x2 = x # f(x)
    df_dx1 = 4 * (x1 - 2)**3 + 2 * (x1 - 2 * x2)
    df_dx2 = -4 * (x1 - 2 * x2)
    return np.array([df_dx1, df_dx2])

a, b = -1, 3
optimal_x, optimal_value = golden_section_method(a, b, EPSILON)
print("Optimal Point:", optimal_value) # 7.16


initial_guess = np.array([0.0, 3.0])
# result, iterations = fletcher_reeves_method(objective_function, gradient, initial_guess) # problem 2d

# Initial interval and accuracy

# print("Optimal Point:", result)
# print("Iterations:", iterations)
