import numpy as np
from scipy.optimize import line_search, golden
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""Using golden section method for computing fletcher reeves"""

def golden_section(f, a, b, epsilon): #univar
    alpha = 0.618  # golden ratio
    varlambda = a + (1 - alpha) * (b - a)  # base case
    varmu = a + alpha * (b - a)  # base case

    while abs(b - a) > epsilon:

        if f(varlambda) > f(varmu):
            a = varlambda
            varlambda = varmu
            varmu = a + alpha * (b - a)

        elif f(varlambda) <= f(varmu):
            b = varmu
            varmu = varlambda
            varlambda = a + (1 - alpha) * (b - a)

    return f((a + b) / 2), (a + b) / 2

def fletcher_reeves(Xj, f, epsilon): # initialization
    x1, x2, NORM = [Xj[0]], [Xj[1]], np.linalg.norm  # initial guesses 
    Df = gradient
    grad_k = gradient(Xj)
    d = - grad_k  
    v_lam = []

    while True:
        start_point = Xj # start point
        v_lambda = golden(lambda lam: f(start_point + lam * d), brack=(a,b), tol=epsilon)
        v_lam.append(v_lambda)

        if v_lambda is not None:
            X = Xj + v_lambda * d # update exp point
            x1.append(X[0])
            x2.append(X[1])

        if NORM(Df(X)) < epsilon:
            return x1, x2, v_lam

        else:
            Xj = X
            temp = grad_k # grad at preceding point
            grad_k = Df(Xj) # grad at current point
            chi = NORM(grad_k)**2/ NORM(temp)**2 
            d = - grad_k + chi*d # new updated descent direction
            x1.append(Xj[0])
            x2.append(Xj[1])
def f(x):
    return  (x[0] - 2*x[1])**2 + (x[0] - 2)**4

def gradient(x):
    df_dx1 = 2*x[0] - 4*x[1] + 4*(x[0]-2)**3 
    df_dx2 = 8*x[1] - 4*x[0]
    return np.array([df_dx1, df_dx2])

#def fletcher_reeves(Xj, epsilon, alpha_1, alpha_2):
a, b, epsilon = -1, 11, 0.02
init_guess = Xj = np.array([0.0, 3.0])
x1, x2, v_lam = fletcher_reeves(init_guess, f, epsilon) # problem 2d

#Initial interval and accuracy

print("X1: ", x1)
print("X2: ", x2)
print("Lambda: ", v_lam)

# Plots
contour_x = np.linspace(0, 3, 100)
contour_y = np.linspace(0, 3, 100)
X, Y = np.meshgrid(contour_x, contour_y)
Z = f([X, Y])

plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=10, cmap='viridis')
plt.scatter(x1[::3], x2[::3], c='blue', marker='x', label='Optimization Path')
plt.plot(x1[::3], x2[::3], linestyle='-', color='k', alpha=1)
plt.title('Optimization Contour Plot')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(contour, label='Function Value')
plt.legend()
plt.show()