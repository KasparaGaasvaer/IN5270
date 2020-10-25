import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class FiniteElementSolverP2():
    def __init__(self, f, N_elements, C, D, analytical, grid_points):
        """ Initialize class variables and matrices/vectors of size according to number of elements"""

        self.gp = grid_points

        self.Ne = N_elements
        self.C = C
        self.D = D
        self.f = lambda x: f(x)
        self.tol = 10e-3
        self.x = sym.Symbol("x")

        self.h = 1/(2*self.Ne)
        self.global_matrix = np.zeros([2*self.Ne, 2*self.Ne])
        self.global_vector = np.zeros([2*self.Ne])
        self.psi = sym.zeros(3*self.Ne,1)

        self.analytical = lambda x,C,D: analytical(x,C,D)

        self.x_values = np.linspace(0,1,self.gp)

    def make_matrix(self):
        """ Calls all methods to produce the lagrange polynomials, the element matrix and right hand side vector."""
        self.leftmost_element()
        self.rightmost_element()
        self.interior_element()

        #Transforms all sympy symbolic expressions for the lagrange polynomials into callable functions.
        self.psi_funcs = [sym.lambdify([self.x], self.psi[i], modules = "numpy") for i in range(3*self.Ne)]

    def find_coefficients(self):
        """ Calls mathod to produce element matrix and vector and computes the coefficients. """
        self.make_matrix()
        self.coeffs = np.linalg.solve(self.global_matrix,self.global_vector)
        self.coeffs = np.append(self.coeffs, self.D)  #Initial condition


    def leftmost_element(self):
        """ Find element matrix and vector for the special case of the leftmost element """


        #Element limits
        L = 0
        R = 2*self.h

        psi0 = (self.x-self.h)*(self.x-2*self.h)/(2*self.h**2)
        psi1 = -self.x*(self.x-2*self.h)/(self.h**2)
        psi2 = self.x*(self.x-self.h)/(2*self.h**2)

        self.psi[0] = psi0
        self.psi[1] = psi1
        self.psi[2] = psi2

        d_psi0 = sym.diff(psi0,self.x)
        d_psi1 = sym.diff(psi1,self.x)
        d_psi2 = sym.diff(psi2,self.x)

        psi_00 = d_psi0*d_psi0
        psi_11 = d_psi1*d_psi1
        psi_22 = d_psi2*d_psi2
        psi_01 = d_psi0*d_psi1
        psi_02 = d_psi0*d_psi2
        psi_12 = d_psi1*d_psi2

        A_00 = sym.integrate(psi_00, (self.x, L, R))
        A_11 = sym.integrate(psi_11, (self.x, L, R))
        A_22 = sym.integrate(psi_22, (self.x, L, R))
        A_01 = sym.integrate(psi_01, (self.x, L, R))
        A_02 = sym.integrate(psi_02, (self.x, L, R))
        A_12 = sym.integrate(psi_12, (self.x, L, R))

        rhs_0 = sym.integrate(self.f(self.x)*psi0,(self.x,L,R)) - self.C
        rhs_1 = sym.integrate(self.f(self.x)*psi1,(self.x,L,R))
        rhs_2 = sym.integrate(self.f(self.x)*psi2,(self.x,L,R))

        a1 = [A_00,A_01,A_02]
        a2 = [A_01, A_11, A_12]
        a3 = [A_02, A_12, A_22]

        A = np.array([a1, a2, a3]).reshape(3,3)#Dette kan gjøres utenfor bro.
        b = np.array([rhs_0, rhs_1, rhs_2])


        for i in range(3):
            self.global_vector[i] = b[i]
            for j in range(3):
                self.global_matrix[i,j] = A[i,j]

    def interior_element(self):
        """ Find element matrix and vector for all interior elements."""

        temp = 0
        for j in range(3,2*self.Ne-2,2):

            L = (j-1)*self.h
            R = (j+1)*self.h

            psi_jm1 = (self.x-self.h*j)*(self.x-self.h*j-self.h)/(2*self.h**2)
            psi_j= -(self.x-self.h*j-self.h)*(self.x-self.h*j+self.h)/(self.h**2)
            psi_jp1 = (self.x-self.h*j)*(self.x-self.h*j+self.h)/(2*self.h**2)

            self.psi[j + temp] = psi_jm1
            self.psi[j + temp + 1] = psi_j
            self.psi[j + temp + 2] = psi_jp1

            d_psi_jm1 = sym.diff(psi_jm1,self.x)
            d_psi_j = sym.diff(psi_j,self.x)
            d_psi_jp1 = sym.diff(psi_jp1,self.x)

            psi_jj = d_psi_j*d_psi_j
            psi_jm1jm1 = d_psi_jm1*d_psi_jm1
            psi_jp1jp1 = d_psi_jp1*d_psi_jp1
            psi_jjm1 = d_psi_j*d_psi_jm1
            psi_jjp1 = d_psi_j*d_psi_jp1
            psi_jm1jp1 = d_psi_jm1*d_psi_jp1

            A_jj = sym.integrate(psi_jj, (self.x, L, R))
            A_jm1jm1 = sym.integrate(psi_jm1jm1, (self.x, L, R))
            A_jp1jp1 = sym.integrate(psi_jp1jp1, (self.x, L, R))
            A_jjm1 = sym.integrate(psi_jjm1, (self.x, L, R))
            A_jjp1 = sym.integrate(psi_jjp1, (self.x, L, R))
            A_jm1jp1 = sym.integrate(psi_jm1jp1, (self.x, L, R))

            rhs_jm1 = sym.integrate(self.f(self.x)*psi_jm1,(self.x,L,R))
            rhs_j = sym.integrate(self.f(self.x)*psi_j,(self.x,L,R))
            rhs_jp1 = sym.integrate(self.f(self.x)*psi_jp1,(self.x,L,R))

            a1 = [A_jm1jm1,A_jjm1,A_jm1jp1]
            a2 = [A_jjm1, A_jj, A_jjp1]
            a3 = [A_jm1jp1, A_jjp1, A_jp1jp1]

            A = np.array([a1, a2, a3]).reshape(3,3)
            b = np.array([rhs_jm1, rhs_j, rhs_jp1])

            for i in range(j-1,j+2):
                self.global_vector[i] += b[i-(j-1)]
                for k in range(j-1,j+2):
                    self.global_matrix[i,k] += A[i-(j-1),k-(j-1)]

            temp += 1

    def rightmost_element(self):
        """ Find element matrix and vector for the special case of the rightmost element """


        #Element limits
        L = 1 - 2*self.h
        R = 1

        psiN = (self.x-1+self.h)*(self.x-1+2*self.h)/(2*self.h**2)
        psiNm1 = -(self.x-1)*(self.x-1+2*self.h)/(self.h**2)
        psiNm2 = (self.x-1+self.h)*(self.x-1)/(2*self.h**2)

        self.psi[-1] = psiN
        self.psi[-2] = psiNm1
        self.psi[-3] = psiNm2

        d_psiNm2 = sym.diff(psiNm2,self.x)
        d_psiNm1 = sym.diff(psiNm1,self.x)
        d_psiN = sym.diff(psiN,self.x)

        psi_Nm1Nm1 = d_psiNm1*d_psiNm1
        psi_Nm2Nm2 = d_psiNm2*d_psiNm2
        psi_Nm1Nm2 = d_psiNm1*d_psiNm2

        A_Nm1Nm1 = sym.integrate(psi_Nm1Nm1, (self.x, L, R))
        A_Nm2Nm2 = sym.integrate(psi_Nm2Nm2, (self.x, L, R))
        A_Nm1Nm2 = sym.integrate(psi_Nm1Nm2, (self.x, L, R))

        rhs_Nm2 = sym.integrate(self.f(self.x)*psiNm2,(self.x,L,R)) - sym.integrate(self.D*d_psiN*d_psiNm2,(self.x,L,R))
        rhs_Nm1 = sym.integrate(self.f(self.x)*psiNm1,(self.x,L,R)) - sym.integrate(self.D*d_psiN*d_psiNm1,(self.x,L,R))

        a1 = [A_Nm2Nm2,A_Nm1Nm2]
        a2 = [A_Nm1Nm2, A_Nm1Nm1]

        A = np.array([a1, a2]).reshape(2,2)
        b = np.array([rhs_Nm2, rhs_Nm1])

        for i in range(2*self.Ne-2,2*self.Ne):
            self.global_vector[i] = b[i-(2*self.Ne-2)]
            for j in range(2*self.Ne-2,2*self.Ne):
                self.global_matrix[i,j] = A[i-(2*self.Ne-2), j-(2*self.Ne-2)]

    def u(self,x,i,temp):
        """Method that returns the approximated values for u(x) in a specified element."""
        return self.coeffs[i]*self.psi_funcs[i+temp](x) + self.coeffs[i+1]*self.psi_funcs[i+temp+1](x) + self.coeffs[i+2]*self.psi_funcs[i+temp+2](x)

    def calculate_numerical_solution(self):
        """ Calculates the approximated solution u(x) over entire grid."""

        self.u = np.vectorize(self.u)
        temp = 0
        self.numerical = np.zeros(self.gp)   #Store value of u(x) in each gridpoint

        #Calculate value of u(x) in each element.
        for i in range(0,2*self.Ne,2):

            L = i* self.h
            R = L + 2*self.h

            #Find idices in x array to define element start/end
            start = np.where((np.abs(self.x_values - L)<=self.tol))[0][0]
            end = np.where((np.abs(self.x_values - R)<=self.tol))[0][0]

            #We want to find L < X <= R, except from first element where we want to include L=0.
            if i != 0:
                start += 1
            end += 1
            x = self.x_values[start:end]
            self.numerical[start:end] = self.u(x,i,temp)

            temp+=1

    def plot_solution(self):
        """Method to plot analytical vs. numerical solution. """

        plt.plot(self.x_values, self.analytical(self.x_values, self.C,self.D), label = "Analytical")
        plt.plot(self.x_values, self.numerical, label = "Numerical")
        plt.title("Numerical vs. Analytical Solution")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.show()


    def automatic_results(self):
        """ Method that calls all other relevant methods to produce
        numerical solution, plot to compare analytical to numerical solution and L2-norm of the error.
        """
        self.find_coefficients()
        self.calculate_numerical_solution()
        self.plot_solution()

    def L2_norm(self):
        """Calculates the L2-norm of the error."""
        analyticals = self.analytical(self.x_values, self.C, self.D)
        error = analyticals - self.numerical
        self.L2 = np.sum(error**2)
