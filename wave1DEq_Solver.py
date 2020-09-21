import numpy as np
import matplotlib.pyplot as plt


class wavesolver1D:
    def __init__(self, h, L, c, T):

        self.h = h
        self.c = c
        self.C = 0.5
        self.CC = self.C*self.C
        self.L = L
        self.T = T
        self.dt = self.h
        self.dx = self.h*self.c/self.C
        self.Nx = round(self.L/self.dx)
        self.u_new = np.zeros(self.Nx)
        self.u = np.zeros(self.Nx)
        self.u_old = np.zeros(self.Nx)
        self.Nt = self.T/self.dt
        self.x = np.linspace(0, self.L, self.Nx)

        self.l2_norm = 0
        self.linf_norm = linf_norm


    def SetAllConditions(self,I):
        self.u_old = I(self.x)
        self.u_old[0] = 0.
        self.u_old[-1] = 0.
        self.u_new[0] = 0.
        self.u_new[-1] = 0.

    #Calculating the first step, working around the fact that we dont have n-1 point for n = 0
    def FirstStep(self):
        for i in range(1,self.Nx-1):
            self.u[i] = self.u_old[i] + 0.5*self.CC*(self.u_old[i+1] - 2*self.u_old[i] + self.u_old[i-1])

    def Advance(self):
        for i in range(1,self.Nx-1):
            self.u_new[i] = -self.u_old[i] + 2*self.u[i] + self.CC*(self.u[i+1] - 2*self.u[i] + self.u[i-1])

    def Swap(self):
        self.u_old[:] = self.u[:]
        self.u[:] = self.u_new[:]

    def TimeEvolution(self,analytical):
        self.t = 2*self.dt
        self.FirstStep()
        while self.t <= self.T:
            self.Advance()
            self.Swap()
            self.Error(analytical)
            self.t += self.dt
        self.l2_norm = np.sqrt(self.dt*self.dx*self.l2_norm)  #For space-time
        return self.l2_norm, self.linf_norm

    def Plot(self,analytical):
        analytical_values = analytical(self.x,self.T)
        plt.plot(self.x, self.u, label = "Numerical")
        plt.plot(self.x, analytical_values, label = "Analytical")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u(x,T) = Amplitude")
        plt.title("Solution of the wave equation at time t = %f" % self.T)
        plt.show()

    def Error(self,analytical):
        computed_error = analytical(self.x,self.t) - self.u[:]
        self.l2_norm += np.sum(computed_error**2)
        test_global_error = np.abs(computed_error).max()
        self.linf_norm = np.max(test_global_error,self.linf_norm)
