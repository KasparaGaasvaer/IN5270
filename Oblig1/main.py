import numpy as np
import matplotlib.pyplot as plt
import sys
from wave_2D import *

task = sys.argv[1]

if task == "3.1":
    b = 0
    T = 1
    Nx = 100
    Ny = 100
    Lx = 1
    Ly = 1

    def I(x,y):
        return 1

    def V(x,y):
        return 0

    def q(x,y):
        return 1

    def f(x,y,t):
        return 0

    def analytical_solution(x,y,t):
        A = 1
        mx = 1
        my = 1
        w = 1

        kx = (mx*np.pi/Lx)
        ky = (my*np.pi/Ly)

        return A*np.cos(kx*x)*np.cos(ky*y)*np.cos(w*t)

    my_solver = Wave2D(b, T, Lx, Ly, I, V, q, Nx, Ny, f)
    my_solver.set_initial_conditions()
    my_solver.time_evolution()
    my_solver.plot("X", "Y", "2D wave equation with constant solution")
    plt.show()

if task == "3.2":
        b = 0
        T = 1
        Lx = 1
        Ly = 1

        A = 2
        mx = 1
        my = 1
        kx = (mx*np.pi/Lx)
        ky = (my*np.pi/Ly)
        w = np.sqrt((mx*np.pi/Lx)**2 + (my*np.pi/Ly)**2)*1   #As long as q =1 is constant

        def q(x,y):
            return 1

        def f(x,y,t):
            return 0

        def I(x,y):
            return A*np.cos(kx*x)*np.cos(ky*y)

        def V(x,y):
            return 0

        def analytical_solution(x,y,t):
            return A*np.cos(kx*x)*np.cos(ky*y)*np.cos(w*t)

        N = [2**i for i in range(1,7)]
        l_inf =[]
        delta_x = []

        for n in N:

            Nx = n
            Ny = n

            my_solver = Wave2D(b, T, Lx, Ly, I, V, q, Nx, Ny, f)
            my_solver.set_initial_conditions()
            my_solver.time_evolution()
            my_solver.true_error(analytical_solution)
            l_inf.append(my_solver.linf_norm)
            delta_x.append(my_solver.dx)

            if n == N[-1]:
                t = my_solver.t
                my_solver.plot("X", "Y", "2D wave equation with standing wave solution")
                x = my_solver.x
                y = my_solver.y
                X,Y = np.meshgrid(x,y)
                plt.contourf(X,Y, analytical_solution(X,Y,t))
                plt.title("Analytical Solution")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.colorbar()
                plt.show()


        r = []
        for i in range(len(l_inf)-1):
            r.append(np.log(l_inf[i+1]/l_inf[i])/np.log(delta_x[i+1]/delta_x[i]))

        print(r)

if task == "3.3":

    A = 1
    w = 1
    b = 2*w

    T = 1
    Lx = 1
    Ly = 1

    mx = 1
    my = 1
    kx = (mx*np.pi/Lx)
    ky = (my*np.pi/Ly)
    K2 = kx**2 + ky**2

    def q(x,y):
        return y

    def analytical_solution(x,y,t):
        return A*np.cos(w*t)*np.exp(-w*t)*np.cos(kx*x)*np.cos(ky*y)

    def I(x,y):
        return A*np.cos(kx*x)*np.cos(ky*y)

    def V(x,y):
        return -w*A*np.cos(kx*x)*np.cos(ky*y)

    def f(x,y,t):
        return A*np.cos(w*t)*np.exp(-w*t)*np.cos(kx*x)*np.cos(ky*y)*(K2*y + ky*np.tan(y*ky)-2*w*w)

    N = [2**i for i in range(1,9)]
    l_inf =[]
    delta_x = []

    for n in N:

        print("N = %i" %n)
        Nx = n
        Ny = n

        my_solver = Wave2D(b, T, Lx, Ly, I, V, q, Nx, Ny, f)
        my_solver.set_initial_conditions()
        my_solver.time_evolution()
        my_solver.true_error(analytical_solution)
        l_inf.append(my_solver.linf_norm)
        delta_x.append(my_solver.dx)


        if n == N[-1]:
            t = my_solver.t
            x = my_solver.x
            y = my_solver.y
            X,Y = np.meshgrid(x,y)
            my_solver.plot("X", "Y", "2D wave with damping and variable wave velocity")
            plt.contourf(X,Y, analytical_solution(X,Y,t))
            plt.title("Analytical Solution")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.colorbar()
            plt.show()

    r = []
    for i in range(len(l_inf)-1):
        r.append(np.log(l_inf[i+1]/l_inf[i])/np.log(delta_x[i+1]/delta_x[i]))

    print(r)

if task == "3.4":

    T = 1
    Lx = 1
    Ly = 1
    g = 9.81

    Im = 0.5 #Peak in the middle of the plot of Im = 0.5
    I0 = 1.
    Ia = 1
    Is = 0.1
    H0 = 10
    B0 = 0.
    Ba = 9
    Bmx = 0.5
    Bmy = 0.5
    Bs = 0.1
    b = 1  # Now a scaling parameter

    def Gaussian_2D(x,y):
        BigB = B0 + Ba*np.exp(-((x-Bmx)/Bs)**2 - ((y-Bmy)/(b*Bs))**2)
        BigH = H0 - BigB
        return g*BigH

    def Cosine_hat(x,y):
        Bs = 0.1 + np.sqrt(Lx**2 + Ly**2)  # Make sure Bs is greater than the restriction
        BigB = B0 + Ba*np.cos(np.pi*(x-Bmx)/(2*Bs))*np.cos(np.pi*(y-Bmy)/(2*Bs))
        BigH = H0 - BigB
        return g*BigH


    def Box(x,y):
        Bmx = Lx
        Bmy = Lx   #Works as long as Lx = Ly
        Bs = Lx   #And b >= 1

        BigB = B0 + Ba
        BigH = H0 - BigB
        return g*BigH


    def I(x,y):
        return I0 + Ia*np.exp(-((x-Im)/Is)**2)

    def V(x,y):
        return 0

    def f(x,y,t):
        return 0

    n = 50

    Nx = n
    Ny = n

    T = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]

    for t in T:
        my_solver = Wave2D(b, t, Lx, Ly, I, V,Gaussian_2D, Nx, Ny, f)
        my_solver.set_initial_conditions()
        my_solver.time_evolution()
        my_solver.plot("X", "Y", " 2D Gaussian hill for T = %f" % t)

    plt.show()
