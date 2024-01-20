import numpy as np

def golden_section_search(f, a, b, epsilon):
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

def fletcher_reeves_method(f, x1, x2, epsilon, a, b):
    y1 = x1 # base cases
    grad_k = gradient([x1, x2])
    d = - grad_k  

    for k in range(100_000):
        _, varlambda = golden_section_search(lambda x: f([x, x2]), a, b, epsilon)

        y1 = y1 + varlambda * d 
        grad_k1 = gradient([x1, x2]) # fix gradient calcs

        alpha = np.dot(grad_k1, grad_k1) / np.dot(grad_k, grad_k) 

        d = -grad_k1 + alpha * d  

        if np.linalg.norm(grad_k1) < epsilon:
            break  

        grad_k = grad_k1  # update me

    return [x1, x2], k+1 # result and iterations

def f(x):
    return (x[0] - 2)**4 + (x[0] - 2 * x[1])**2

def gradient(x):
    df_dx1 = 4 * (x[0] - 2)**3 + 2 * (x[0] - 2 * x[1])
    df_dx2 = -4 * (x[0] - 2 * x[1])
    return np.array([df_dx1, df_dx2])

a, b, epsilon = 0, 3, 0.02

init = np.array([0.0, 3.0])
result, iterations = fletcher_reeves_method(f, init[0], init[1], epsilon, a, b) # problem 2d

#Initial interval and accuracy

print("Optimal Point:", result)
print("Iterations:", iterations)
