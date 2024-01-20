import numpy as np
from scipy.optimize import line_search, golden
import matplotlib.pyplot as plt

#def hooke_jeeves(f, x_init, sigma_init, epsilon, alpha):
def hooke_jeeves(f, initial_point, initial_step_size, epsilon, acceleration_factor):
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

def golden_section_search(f, initial_guess, direction, a, b, epsilon):
    # Golden section search to find the minimum along the direction
    v_lambda = golden(lambda lam: f(initial_guess + lam * direction), brack=(a, b), tol=epsilon)
    return initial_guess + v_lambda * direction, v_lambda

def f(x):
    return  (x[0] - 2*x[1])**2 + (x[0] - 2)**4

a, b, epsilon, alpha = -1, 11, 0.02, 0.05
initial_step_size = 5
init_guess = Xj = np.array([0.0, 3.0])
trajectory = hooke_jeeves(f, Xj, initial_step_size, epsilon, alpha)

print("Optimal solution:", trajectory)
print("Optimal value:", f(trajectory))


plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label='Trajectory')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Hooke-Jeeves Optimization Trajectory')
plt.legend()
plt.show()