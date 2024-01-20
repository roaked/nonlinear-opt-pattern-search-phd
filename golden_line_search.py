import math
import matplotlib.pyplot as plt

def golden_section_search(f, a, b, tol):
    alpha = 0.618  # golden ratio
    varlambda = a + (1 - alpha) * (b - a)  # base case
    varmu = a + alpha * (b - a)  # base case

    a_values = [a]
    b_values = [b]
    varlambda_values = [varlambda]
    varmu_values = [varmu]

    while b - a > tol:

        if f(varlambda) > f(varmu):
            a = varlambda
            varlambda = varmu
            varmu = a + alpha * (b - a)

        elif f(varlambda) <= f(varmu):
            b = varmu
            varmu = varlambda
            varlambda = a + (1 - alpha) * (b - a)

        a_values.append(a)
        b_values.append(b)
        varlambda_values.append(varlambda)
        varmu_values.append(varmu)

    return f((a + b) / 2), (a + b) / 2, a_values, b_values, varlambda_values, varmu_values


def f(x1):
    return 6 * (math.e) ** (-2 * x1) + 2 * x1**2

a = -1
b = 3
tol = 0.002
min_value, min_x, a_values, b_values, varlambda_values, varmu_values = golden_section_search(f, a, b, tol)

fig, ax = plt.subplots()
iterations = range(len(a_values))

ax.plot(iterations, a_values, label='a')
ax.plot(iterations, b_values, label='b')
ax.plot(iterations, varlambda_values, label='varlambda')
ax.plot(iterations, varmu_values, label='varmu')

ax.set_xlabel('Iterations')
ax.set_ylabel('Values')
ax.legend()
plt.show()
