import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D

'''
Solving Poisson's and Laplace equations
Marco Mas Sempere
'''

'''
TridiagonalSolver for solving 1D problems
'''

def TridiagonalSolver(d,o,u,r):
    n = len(d)
    x = np.zeros(n)

    if n  == len(o) + 1 == len(u) + 1:       #Compruebo que cumple las condiciones que debe para poder resolverlo
        try:
            for i in range(n-1):      #Hago unos en la diagonal
                o[i],r[i] = o[i]/d[i],r[i]/d[i] 
                d[i] = 1
                d[i+1],r[i+1] = d[i+1] - o[i]*u[i],r[i+1] - r[i]*u[i]


            x[-1] = r[-1]/d[-1]       #Calculamos la solución de X[N-1]
            for i in range(2,n+1):      #Calculamos el resto de soluciones
                x[-i] = r[-i] - o[-i+1] * x[-i+1]
            return x
        except:
            print('Cero en algun elemento de la diagonal')
    
    else:
        print('Las dimensiones no son correctas')




'''
Part 1
EDO f''(x) = g(x)
Poisson Equation in 1D
'''



#Defining g(x) for 1 example
def g(x):
    return (2-x**2)*np.sin(x) + 4*x*np.cos(x)


#Discretization of the domain

N = 1000
xi = np.linspace(0,np.pi,N)
h = xi[1]-xi[0]



#Making the matrix of coefficients, taking advantage of its properties

diag = -2*np.ones(N)
over = np.ones(N-1)
under = np.ones(N-1)
gi = (h**2) * g(xi)



#Boundary conditions

over[0] = 0
diag[0] = 1
gi[0] = 0

under[-1] = -1
diag[-1] = 1
gi[-1] = -h*np.pi**2



#Solving the system

fi = TridiagonalSolver(diag, over, under, gi)



#Plotting the solution

plt.figure()
plt.title('Solution of the EDO $ \\frac{d^2f}{dx^2} = (2-x^2) \sin{x} + 4x \cos{x}$')
plt.plot(xi,fi)
plt.ylabel('f(x)')
plt.xlabel('x')



 
'''
Part 2
EDP $\\nabla^2 f(x,y) = g(x,y)$
Laplace Equation in 2D
'''



#Discretization of the domain
N = 100

xi = np.linspace(0,1,N)
yi = np.linspace(0,1,N)

delta = xi[1] - xi[0]

X, Y =  np.meshgrid(xi,yi)



#Making the matrix of coefficients, taking advantage of its properties


#Primero hacemos los bloques de la diagonal
#First we make the blocks of the diagonal

diag = -4*np.ones(N)
diag[0] = 1
diag[-1] = 1

o = np.ones(N-1)
o[0] = 0

u = np.ones(N-1)
u[-1] = 0

D = sparse.diags([diag, o, u], [0,1,-1],shape=(N,N))



#Now we make the identity matrix and the blocks of the off-diagonal

I = sparse.eye(N)

o1 = np.ones(N)
o2 = np.ones(N)

o1[0] = 0
o1[-1] = 0

o2 = o2 - o1

I1 = sparse.diags([o1], [0],shape=(N,N))
I2 = sparse.diags([o2], [0],shape=(N,N))


o3 = np.ones(N-1)
o3[0] = 0

o4 = np.ones(N-1)
o4[-1] = 0

I3 = sparse.diags([o3,o4], [1,-1],shape=(N,N))


A = sparse.kron(I2,I) + sparse.kron(I1,D) + sparse.kron(I3,I1)

gi = np.zeros(N**2)
gi[N**2-N:N**2]=xi*(1-xi)






#Resolvemos el sistema y reordenamos la solución en forma de matriz

fij = sparse.linalg.spsolve(A, gi).reshape((N,N))



#Making the 3D plot of the solution

fig = plt.figure('Plot in 3D of the solution to Laplace equation')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, fij, cmap='RdBu', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



#Making the 2D plot of the solution

plt.figure()
plt.title('Plot in 2D of the solution to Laplace equation')
plt.imshow(fij[:][::-1],extent=[0, 1, 0, 1])
plt.xlabel('X')
plt.ylabel('Y')








'''
Part 3
Iterative Method (Jacobi)
'''



#Discretization of the domain

N = 100
xi = np.linspace(0,1,N)
yi = np.linspace(0,1,N)

X, Y =  np.meshgrid(xi,yi)



#Initial guess for the solution, with the boundary conditions already applied

phi0 = np.zeros(N**2)
phi0[N**2-N:N**2]=xi*(1-xi)



#Making the matrix of coefficients

o1 = 0.25*np.ones(N**2-1)
o2 = 0.25*np.ones(N**2-N)
D = sparse.diags([o1,o1,o2,o2],[1,-1,N,-N],shape=(N**2,N**2))



#First iteration, applying the boundary conditions

phi1 = D.dot(phi0).reshape(N,N)
phi1[0,:],phi1[0,:],phi1[:,0],phi1[:,1]= np.zeros(N),xi*(1-xi),np.zeros(N),np.zeros(N)
phi1 = phi1.reshape(N**2)





#Iterating until convergence, applying the boundary conditions at each step

tol = 10**(-4)

while np.sum(np.abs(phi1-phi0)) > tol:
    phi0 = phi1
    phi1 = D.dot(phi0).reshape(N,N)
    phi1[0,:],phi1[-1,:],phi1[:,0],phi1[:,-1]= np.zeros(N),xi*(1-xi),np.zeros(N),np.zeros(N)
    phi1 = phi1.reshape(N**2)



#Making the 3D plot of the solution

phi1 = phi1.reshape(N,N)
fig = plt.figure('Plot in 3D of the solution to Laplace equation with Jacobi method')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phi1, cmap='RdBu', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
