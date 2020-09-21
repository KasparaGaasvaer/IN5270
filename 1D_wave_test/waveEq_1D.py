import numpy as np
from wave1DEq_Solver import wavesolver1D

L = 1
c = 1
T = 1

exps = 8

def I(x):
    "A function for the form of the string"
    return np.sin(2*np.pi*x/L)

def analytical_solution(x,t):
    return np.sin(2*np.pi*x/L)*np.cos(2*np.pi*c*t/L)

h0 = 0.05 #For stor h, typ h = 0.5, gir zero-div på grunn av log senere
h = [h0*2**(-i) for i in range(exps)]
l2 = []
linf_norm = []
}

for hi in h:
    my_solver = wavesolver1D(hi,L,c,T)
    my_solver.SetAllConditions(I)
    l2_norm, linf_norm = my_solver.TimeEvolution(analytical_solution)
    l2.append(l2_norm)
    print(hi)
    #my_solver.Plot(analytical_solution)

r = []

for i in range(exps-1):
    r.append(np.log(l2[i+1]/l2[i])/np.log(h[i+1]/h[i]))


print(r)
