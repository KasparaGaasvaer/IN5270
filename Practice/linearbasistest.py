import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2,1,0,0],[1,2,0,0],[0,0,2,1],[0,0,1,2]]).reshape(4,4)/12
b = np.array([31,37,45,47])/96

X = np.linalg.solve(A,b)

print(X)


def interpol(x, middle):

    if x <= middle:
        phi0 = -2*x + 1
        phi1 = 2*x
        return X[0]*phi0 + X[1]*phi1

    if middle < x:
        phi2 = -2*x + 2
        phi3 = 2*x - 1
        return X[2]*phi2 + X[3]*phi3



x = np.linspace(0,1,1001)

def f(x):
    return 1 + 2*x - x**2

f = np.vectorize(f)
interpol = np.vectorize(interpol)

plt.plot(x,f(x), label = "True function")
plt.plot(x, interpol(x ,0.5), label = "Interpolating polynomial")
plt.legend()
plt.show()
