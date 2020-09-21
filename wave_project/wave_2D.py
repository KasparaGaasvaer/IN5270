import numpy as np
import matplotlib.pyplot as plt


class Wave2D():
    def __init__(self, b, T, Lx, Ly, I, V, qq, Nx, Ny, f):
        """ Initializing class variables and functions  """

        self.b = b
        self.Nx = Nx
        self.Lx = Lx
        self.Ly = Ly
        self.Ny = Ny
        self.x = np.linspace(0,self.Lx,self.Nx)
        self.y = np.linspace(0,self.Ly,self.Ny)
        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.X = self.X.T
        self.X = self.Y.T
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Making functions sent to init availible for all methods in class
        self.f = lambda x,y,t: f(x,y,t)
        self.I = lambda x,y: I(x,y)
        self.V = lambda x,y: V(x,y)
        self.qq = lambda x,y: qq(x,y)
        self.make_q()

        self.T = T
        self.Nt = int(round(self.T/self.dt))

        #Defining arrays to hold solutions
        self.u_new = np.zeros((self.Nx+2, self.Ny+2))
        self.u = np.zeros((self.Nx+2, self.Ny+2))
        self.u_old = np.zeros((self.Nx+2, self.Ny+2))

        #Variables for simplyfying the mathematical expressions
        self.dt2 = self.dt*self.dt
        self.E = 1./self.dt2
        self.B = self.b/(2*self.dt)
        self.Cy = 1./(2*self.dy*self.dy)
        self.Cx = 1./(2*self.dx*self.dx)


        # Variable for holding error norm
        self.linf_norm = 0

    def stability(self):
        """ Sets dt after stability criteria"""

        maximum_velocity = np.max(self.q[1:self.Nx+1,1:self.Ny+1])
        beta_factor = 0.9
        self.dt = beta_factor*(1./np.sqrt(maximum_velocity))*(1/np.sqrt(1/self.dx**2 + 1/self.dy**2))

    def set_initial_conditions(self):
        """ Setting initial values and solving first modified step """

        self.set_initial_u_old()
        self.first_step()

    def set_initial_u_old(self):
        """ Initializes the first u^(n-1) on the inner mesh points/boundary
        and calls a method to initialize ghost cells as well"""

        for i in range(1, self.Nx +1):
            for j in range(1, self.Ny +1):
                self.u_old[i,j] = self.I(self.x[i-1], self.y[j-1])

        self.updating_ghost_cells(self.u_old)

    def updating_ghost_cells(self,uu):
        """ Method for updating ghost cells """


        for i in range(1, self.Nx +1):
            uu[i,0] = uu[i,2]
            uu[i,self.Ny+1] = uu[i,self.Ny-1]


        for j in range(1, self.Ny +1):
            uu[0,j] = uu[2,j]
            uu[self.Nx+1,j] = uu[self.Nx-1,j]
        """

        uu[1:-1,0] = uu[1:-1,2]
        uu[1:-1,self.Ny+1] = uu[1:-1,self.Ny-1]
        uu[0,1:-1] = uu[2,1:-1]
        uu[self.Nx+1,1:-1] = uu[self.Nx-1,1:-1]
        """
    def make_q(self):
        """ Fills a matrix with values from the q(x,y) function """
        self.q = np.zeros((self.Nx+2,self.Ny+2))


        for i in range(1, self.Nx+1):
            for j in range(1, self.Ny+1):
                self.q[i,j] = self.qq(self.x[i-1], self.y[j-1])

        for i in range(1,self.Nx+1):
            self.q[i,0] = 2*self.q[i,1] - self.q[i,2]
            self.q[i,self.Ny +1] = 2*self.q[i,self.Ny] - self.q[i,self.Ny-1]


        for j in range(1,self.Ny+1):
            self.q[0,j] = 2*self.q[1,j] - self.q[2,j]
            self.q[self.Nx+1,j] = 2*self.q[self.Nx,j] - self.q[self.Nx-1,j]

        """

        self.q[1:-1, 1:-1] = self.qq(self.X, self.Y)
        self.q[1:-1, 0] = 2*self.q[1:-1, 1] - self.q[1:-1, 2]
        self.q[1:-1, self.Ny +1] = 2*self.q[1:-1, self.Ny] - self.q[1:-1, self.Ny-1]
        self.q[0, 1:-1] = 2*self.q[1, 1:-1] - self.q[2, 1:-1]
        self.q[self.Nx+1, 1:-1] = 2*self.q[self.Nx, 1:-1] - self.q[self.Nx-1, 1:-1]
        """
        self.stability()

    def first_step(self):
        """ Calculates the first modified step and calls method
        to update ghost cells """

        q = self.q
        u_old = self.u_old


        for i in range(1, self.Nx+1):
            for j in range(1, self.Ny+1):

                self.u[i,j] = u_old[i,j] + (1/(2*self.E))*(self.Cx*((q[i,j] + q[i+1,j]) * (u_old[i+1,j] - u_old[i,j]) - (q[i,j] + q[i-1,j]) * (u_old[i,j] - u_old[i-1,j])) + self.Cy*((q[i,j] + q[i,j+1]) * (u_old[i,j+1] - u_old[i,j]) - (q[i,j] + q[i,j-1]) * (u_old[i,j] - u_old[i,j-1])) + 2*self.dt*self.V(self.x[i-1], self.y[j-1])*(self.E - self.B) + self.f(self.x[i-1], self.y[j-1],0))
        """

        self.u[1:-1,1:-1] = u_old[1:-1,1:-1] + (1/(2*self.E))*(self.Cx*((q[1:-1,1:-1] + q[2:,1:-1]) * (u_old[2:,1:-1] - u_old[1:-1,1:-1]) - (q[1:-1,1:-1] + q[0:-2,1:-1]) * (u_old[1:-1,1:-1] - u_old[0:-2,1:-1])) + self.Cy*((q[1:-1,1:-1] + q[1:-1,2:]) * (u_old[1:-1,2:] - u_old[1:-1,1:-1]) - (q[1:-1,1:-1] + q[1:-1,0:-2]) * (u_old[1:-1,1:-1] - u_old[1:-1,0:-2])) + 2*self.dt*self.V(self.X, self.Y)*(self.E - self.B) + self.f(self.X, self.Y,0))
        """
        self.updating_ghost_cells(self.u)

    def advance_general_scheme(self):
        """ The general scheme for advancing the solution"""

        q = self.q
        u = self.u

        for i in range(1, self.Nx+1):
            for j in range(1, self.Ny+1):

                self.u_new[i,j] = (1/(self.E+self.B))*(self.Cx*((q[i,j] + q[i+1,j])*(u[i+1,j] - u[i,j]) - (q[i,j] + q[i-1,j]) * (u[i,j] - u[i-1,j])) + self.Cy*((q[i,j] + q[i,j+1]) * (u[i,j+1] - u[i,j]) - (q[i,j] + q[i,j-1]) * (u[i,j] - u[i,j-1])) + 2*self.E*u[i,j] - (self.E - self.B) * self.u_old[i,j] + self.f(self.x[i-1], self.y[j-1], self.t))
        """

        self.u_new[1:-1,1:-1] = (1/(self.E+self.B))*(self.Cx*((q[1:-1,1:-1] + q[2:,1:-1])*(u[2:,1:-1] - u[1:-1,1:-1]) - (q[1:-1,1:-1] + q[0:-2,1:-1]) * (u[1:-1,1:-1] - u[0:-2,1:-1])) + self.Cy*((q[1:-1,1:-1] + q[1:-1,2:]) * (u[1:-1,2:] - u[1:-1,1:-1]) - (q[1:-1,1:-1] + q[1:-1,0:-2]) * (u[1:-1,1:-1] - u[1:-1,0:-2])) + 2*self.E*u[1:-1,1:-1] - (self.E - self.B) * self.u_old[1:-1,1:-1] + self.f(self.X, self.Y, self.t))
        """
        self.updating_ghost_cells(self.u_new)

    def swap(self):
        """ Swaps u variables for each time step"""
        self.u_old, self.u, self.u_new = self.u, self.u_new, self.u_old

    def time_evolution(self):
        """ Method for progressing the time evolution of the solution. """

        self.t = self.dt
        while self.t <= self.T:
            self.advance_general_scheme()
            self.swap()
            self.t += self.dt

    def plot(self, X_ax, Y_ax , title):
        """ Plots final solution in the X,Y-plane with amplitude of wave as contour"""

        plt.contourf(self.X.T,self.Y.T, self.u[1:self.Nx+1,1:self.Ny+1])
        plt.title(title)
        plt.xlabel(X_ax)
        plt.ylabel(Y_ax)
        plt.colorbar()
        plt.figure()




    def true_error(self,analytical):
        """ Calculates the linf norm"""

        analytical_values = np.zeros([self.Nx,self.Ny])
        analytic = lambda x,y,t: analytical(x,y,t)

        for i in range(0,self.Nx):
            for j in range(0,self.Ny):
                analytical_values[i,j] = analytic(self.x[i],self.y[j],self.t)

        computed_error = analytical_values - self.u[1:self.Nx+1, 1:self.Ny+1]
        self.linf_norm = np.max(np.abs(computed_error))
