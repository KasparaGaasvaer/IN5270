import numpy as np
import matplotlib.pyplot as plt
from p2_finite_elem import FiniteElementSolverP2
import sys

Ne = sys.argv[1]

n = 1000
C = 0
D = 1

def f(x):
    return 2*x - 1

def analytical(x,C,D):
    return 0.5*x**2 - (1/3)*x**3 + (x-1)*C + D - (1/6)


if Ne == "L2":

    Ne_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    L2s = []

    for ne in Ne_list:
            print("\n Ne = ", ne)
            my_solver = FiniteElementSolverP2(f, ne, C, D, analytical, n)
            my_solver.find_coefficients()
            my_solver.calculate_numerical_solution()
            my_solver.L2_norm()
            L2s.append(my_solver.L2)
            print("L2-norm = %.16f" %(my_solver.L2))

    Ne_list = np.array(Ne_list)
    L2s = np.array(L2s)

    h = 1/Ne_list


    r = []
    for i in range(len(L2s)-1):
        r.append(np.log(L2s[i+1]/L2s[i])/np.log(h[i+1]/h[i]))

    print(r)

else:

    Ne = int(Ne)
    my_solver = FiniteElementSolverP2(f, Ne, C, D, analytical, n)
    my_solver.automatic_results()
    print("L2-norm = %.16f" %(my_solver.L2))
