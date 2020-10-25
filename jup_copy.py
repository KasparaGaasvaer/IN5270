import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from fractions import Fraction

x = sym.Symbol("x")
Ne = 3
Nn = 2*Ne
C = 0
D = 0
j = 3
h = 1/(Nn)

f = 2*x - 1

""" RIGHTMOST ELEMENT """
psiNm2 = (x-1+h)*(x-1)/(2*h**2)
psiNm1 = -(x-1)*(x-1+2*h)/(h**2)
psiN = (x-1+h)*(x-1+2*h)/(2*h**2)


d_psiNm2 = sym.diff(psiNm2,x)
d_psiNm1 = sym.diff(psiNm1,x)
d_psiN = sym.diff(psiN,x)


psi_Nm1Nm1 = sym.simplify(d_psiNm1*d_psiNm1)
psi_Nm2Nm2 = sym.simplify(d_psiNm2*d_psiNm2)
psi_Nm1Nm2 = sym.simplify(d_psiNm1*d_psiNm2)

#Limits
L = 1 - 2*h
R = 1

A_Nm1Nm1 = sym.simplify(sym.integrate(psi_Nm1Nm1, (x, L, R)))
A_Nm2Nm2 = sym.simplify(sym.integrate(psi_Nm2Nm2, (x, L, R)))
A_Nm1Nm2 = sym.simplify(sym.integrate(psi_Nm1Nm2, (x, L, R)))

rhs_Nm2 = sym.integrate(sym.simplify(f*psiNm2),(x,L,R)) - sym.integrate(sym.simplify(D*d_psiN*d_psiNm2),(x,L,R))
rhs_Nm1 = sym.integrate(sym.simplify(f*psiNm1),(x,L,R)) - sym.integrate(sym.simplify(D*d_psiN*d_psiNm1),(x,L,R))


""" LEFTMOST ELEMENT """
#Limits
L = 0
R = 2*h

psi0 = (x-h)*(x-2*h)/(2*h**2)
psi1 = -x*(x-2*h)/(h**2)
psi2 = x*(x-h)/(2*h**2)

d_psi0 = sym.diff(psi0,x)
d_psi1 = sym.diff(psi1,x)
d_psi2 = sym.diff(psi2,x)

psi_00 = sym.simplify(d_psi0*d_psi0)
psi_11 = sym.simplify(d_psi1*d_psi1)
psi_22 = sym.simplify(d_psi2*d_psi2)
psi_01 = sym.simplify(d_psi0*d_psi1)
psi_02 = sym.simplify(d_psi0*d_psi2)
psi_12 = sym.simplify(d_psi1*d_psi2)

A_00 = sym.integrate(psi_00, (x, L, R))
A_11 = sym.integrate(psi_11, (x, L, R))
A_22 = sym.integrate(psi_22, (x, L, R))
A_01 = sym.integrate(psi_01, (x, L, R))
A_02 = sym.integrate(psi_02, (x, L, R))
A_12 = sym.integrate(psi_12, (x, L, R))

rhs_0 = sym.integrate(f*psi0,(x,L,R)) - C
rhs_1 = sym.integrate(f*psi1,(x,L,R))
rhs_2 = sym.integrate(f*psi2,(x,L,R))



""" ARBITRARY INNER ELEMENT """

psi_jm1 = (x-h*j)*(x-h*j-h)/(2*h**2)
psi_j= -(x-h*j-h)*(x-h*j+h)/(h**2)
psi_jp1 = (x-h*j)*(x-h*j+h)/(2*h**2)

d_psi_jm1 = sym.diff(psi_jm1,x)
d_psi_j = sym.diff(psi_j,x)
d_psi_jp1 = sym.diff(psi_jp1,x)


psi_jj = d_psi_j*d_psi_j
psi_jm1jm1 = d_psi_jm1*d_psi_jm1
psi_jp1jp1 = sym.simplify(d_psi_jp1*d_psi_jp1)
psi_jjm1 = sym.simplify(d_psi_j*d_psi_jm1)
psi_jjp1 = sym.simplify(d_psi_j*d_psi_jp1)
psi_jm1jp1 = sym.simplify(d_psi_jm1*d_psi_jp1)

#Limits
L = h*(j-1)
R = h*(j+1)

A_jj = sym.simplify(sym.integrate(psi_jj, (x, L, R)))
A_jm1jm1 = sym.simplify(sym.integrate(psi_jm1jm1, (x, L, R)))
A_jp1jp1 = sym.simplify(sym.integrate(psi_jp1jp1, (x, L, R)))
A_jjm1 = sym.simplify(sym.integrate(psi_jjm1, (x, L, R)))
A_jjp1 = sym.simplify(sym.integrate(psi_jjp1, (x, L, R)))
A_jm1jp1 = sym.simplify(sym.integrate(psi_jm1jp1, (x, L, R)))


rhs_jm1 = sym.integrate(sym.simplify(f*psi_jm1),(x,L,R))
rhs_j = sym.integrate(sym.simplify(f*psi_j),(x,L,R))
rhs_jp1 = sym.integrate(sym.simplify(f*psi_jp1),(x,L,R))


b0 = rhs_0
b1 = rhs_1
b2 = rhs_2 + rhs_jm1
b3 = rhs_j
b4 = rhs_jp1 + rhs_Nm2
b5 = rhs_Nm1

A0 = [7/6,-4/3,1/6,0,0,0]
A1 = [-4/3,8/3,-4/3,0,0,0]
A2 = [1/6,-4/3,7/3,-4/3,1/6,0]
A3 =[0,0,-4/3,8/3,-4/3,0]
A4 = [0,0,1/6,-4/3,7/3,-4/3]
A5 = [0,0,0,0,-4/3,8/3]


A = np.array([A0,A1,A2,A3,A4,A5]).reshape(6,6)*(1/h)
#print(A)
b = np.array([b0,b1,b2,b3,b4,b5],dtype = "float64").reshape(6,1)
#print(b)


X = np.linalg.solve(A,b)
X = np.append(X, D)



def u(x,j):
    psi0 = (x-h)*(x-2*h)/(2*h**2)
    psi1 = -x*(x-2*h)/(h**2)
    psi2 = x*(x-h)/(2*h**2)

    psi_jm1 = (x-h*j)*(x - h*j -h)/(2*h**2)
    psi_j= -(x-h*j -h)*(x-h*j + h)/(h**2)
    psi_jp1 = (x-h*j)*(x-h*j+h)/(2*h**2)

    psiNm2 = (x-1+h)*(x-1)/(2*h**2)
    psiNm1 = -(x-1)*(x-1+2*h)/(h**2)
    psiN = (x-1+h)*(x-1+2*h)/(2*h**2)


    if x <= (1/3):
        return X[0]*psi0 + X[1]*psi1 + X[2]*psi2
    if x >= (2/3):
        return X[4]*psiNm2 + X[5]*psiNm1 + X[6]*psiN
    else:
        return X[2]*psi_jm1 + X[3]*psi_j + X[4]*psi_jp1

u = np.vectorize(u)
x = np.linspace(0,1,1000)

def analytical(x,C,D):
    return 0.5*x**2 - (1/3)*x**3 + (x-1)*C + D - (1/6)


plt.plot(x, analytical(x,C,D), label = "Analytical")
plt.plot(x, u(x,j), label = "Numerical")
plt.title("C = %i, D = %i" % (C,D))
plt.legend()
plt.show()
