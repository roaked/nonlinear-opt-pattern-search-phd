import numpy as np
from scipy.optimize import golden
import matplotlib.pyplot as plt

def f(x):
    return  (x[0] - 2*x[1])**2 + (x[0] - 2)**4

def gradient(x):
    df_dx1 = 2*x[0] - 4*x[1] + 4*(x[0]-2)**3 
    df_dx2 = 8*x[1] - 4*x[0]
    return np.array([df_dx1, df_dx2]) 

def dfp(Xj, epsilon):
    x1, x2 = [Xj[0]], [Xj[1]]
    Bf = np.eye(len(Xj)) # diagonal with 1's
    v_lam = [] # for lambda values

    while True:
        Grad = gradient(Xj)
        d = -Bf.dot(Grad) # direction of the steepest descent   
        start_point = Xj # start point for step length selection 
        v_lambda = golden(lambda lam: f(start_point + lam * d), brack=(a,b), tol=epsilon)
        v_lam.append(v_lambda)

        if v_lambda is not None:
            X = Xj + v_lambda * d

        if NORM(gradient(X)) < epsilon:
            x1.append(X[0]), x2.append(X[1])
            return X, f(X), v_lambda, x1, x2
        
        else:
            Dj = X - Xj 
            Gj = gradient(X) - Grad 
            w1 = Dj 
            w2 = Bf.dot(Gj) 
            w1T, w2T = w1.T, w2.T
            sigma1, sigma2 = 1/(w1T.dot(Gj)), -1/(w2T.dot(Gj)) 
            W1, W2 = np.outer(w1, w1), np.outer(w2, w2)

            Delta = sigma1*W1 + sigma2*W2 
            Bf += Delta
            Xj = X 
            x1.append(X[0]), x2.append(X[1])

NORM = np.linalg.norm
a, b, epsilon, alpha_1, alpha_2 = -1, 11, 0.002, 10**(-4) ,3.82
init_guess = Xj = np.array([0.0, 3.0])
x, fx, v_lam, t_x1, t_x2 = dfp(init_guess, epsilon) # problem 2d

x1 = np.linspace(0, 3, 25)
x2 = np.linspace(0, 3, 25)
z = np.zeros(([len(x1), len(x2)]))
for i in range(0, len(x1)):
    for j in range(0, len(x2)):
        z[j, i] = f([x1[i], x2[j]])

contours = plt.contour(x1, x2, z, 15, cmap=plt.cm.viridis)  # Change cmap to viridis
plt.clabel(contours, inline=1, fontsize=2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.plot(t_x1, t_x2, "rx-", ms=5.5)  # Plot the trajectory
plt.show()