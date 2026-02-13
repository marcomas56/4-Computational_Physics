import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from matplotlib import animation


'''
Heat Equation
Marco Mas Sempere
'''


'''
Part 1
Solving the Heat Equation in 1D
Equation: $\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}$
'''

#Defining a function to animate F(x,t)

def animacion1D(F,x,t,k,titulo) -> None:
    '''
    Function that animates a function F(x,t)
    '''
    Nt = len(t)
    dt = t[1]-t[0]

    fig = plt.figure(titulo)
    fig.suptitle(titulo)
    ax = fig.add_subplot(ylim=(np.min(F),np.max(F)))
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    line,= ax.plot(x ,F[:,0], label = 't = {} s'.format(0))
    L=ax.legend()
    
    def plotsim(i):
        line.set_data(x,F[:,i*k])
        L.get_texts()[0].set_text('t = {} s'.format(round(i*k*dt,1)))
        
        return line
    
    ani = animation.FuncAnimation(fig,plotsim,frames = Nt//k,interval = 20)
    plt.show()






#Defining the constants needed

D = 0.1



#Discretization of the spatial domain

x0 = -1
x1 = 1
nx = 100

xi = np.linspace(x0,x1 ,nx)
dx = xi[1]-xi[0]



#Discretization of the temporal domain

t0 = 0
t1 = 1
nt = 10000

ti = np.linspace(t0,t1,nt)
dt = ti[1]-ti[0]


#Defining the Initial Conditions (IC) function

def CI1D(x) -> np.ndarray:
    '''
    Function that returns the initial conditions
    for the 1D spatial case
    '''
    return 100*np.exp(-20*x**2)


'''
Explicit Method
$u_i^{n+1} = u_i^n + \frac{D \Delta t}{\Delta x^2} (u_{i+1}^n - 2u_i^n + u_{i-1}^n)$
'''


#Generating the diagonals for the matrix discretizing the second derivative

diag = -2*np.ones(nx)
over = np.ones(nx-1)
under = np.ones(nx-1)



#Imposing boundary conditions

diag[0],diag[-1],over[0],under[-1] = 0,0,0,0

A = sparse.diags([diag, over, under], [0,1,-1],shape=(nx,nx))

I = sparse.diags([diag/(-2)], [0],shape=(nx,nx))



#Generating the matrix with initial values

U = np.zeros((nx,nt))

U[:,0] = CI1D(xi)



#Loop for time propagation using the explicit method

A1 = (dt*D)/(dx**2)*A + sparse.eye(nx)

for j in range(nt-1):
    U[:,j+1] = (A1).dot(U[:,j])


animacion1D(U,xi,ti,25,'Temperature evolution using Explicit Method')




'''
Crank-Nicolson Method
Implicit scheme requiring matrix inversion
'''


#Generating the diagonals for the matrix discretizing the second derivative and imposing BCs

diag = -2*np.ones(nx)
over = np.ones(nx-1)
under = np.ones(nx-1)


diag[0],diag[-1],over[0],under[-1] = 0,0,0,0

A = sparse.diags([diag, over, under], [0,1,-1],shape=(nx,nx)).toarray()

I = sparse.diags([diag/(-2)], [0],shape=(nx,nx))

A1 = (-(dt*D)/(2*dx**2)*A + sparse.eye(nx))

A1[-1,-1],A1[-1,-2] = 1/dx + 1,-1/dx

A2 = (dt*D)/(2*dx**2)*A + I


B = sparse.csc_matrix(np.dot(np.linalg.inv(A1),A2))



#Generating the matrix with initial values

U = np.zeros((nx,nt))

U[:,0] = CI1D(xi)



#Loop for time propagation


for j in range(nt-1):
    U[:,j+1] = B.dot(U[:,j])


animacion1D(U,xi,ti,25,'Temperature evolution using Crank-Nicolson Method')





'''
Part 2
Solving the Heat Equation in 2D
Equation: $\frac{\partial u}{\partial t} = D \nabla^2 u$
'''

#Defining a function to animate F(x,y,t)

def animacion2D(F,x,y,t,k,titulo) -> None:
    '''
    Function that animates a function F(x,y,t)
    '''
    Nt = len(t)
    dt = t[1]-t[0]

    fig = plt.figure(titulo)
    fig.suptitle(titulo)

    im = plt.imshow(F[:,:,0], interpolation= "None",extent=[-1, 1, -1, 1])
    plt.xlabel('x')
    plt.ylabel('y')

    cbar = fig.colorbar(im)

    lab = plt.text(0.05, 0.9, 't = {} s'.format(0),color = 'white')
    
    def plotsim(i):
        im.set_array(F[:,:,k*i])
        lab.set_text('t = {} s'.format(round(i*k*dt,1)))
        return [im]
    
    ani = animation.FuncAnimation(fig,plotsim,frames = Nt//k,interval = 20)
    plt.show()


#Defining the Initial Conditions (IC) function

def CI2D(x,y) -> np.ndarray:
    '''
    Function that returns the initial conditions
    for the 2D spatial case
    '''
    return 100*np.exp(-20*(x**2+y**2))

    
#Defining the constants needed

D = 0.1



#Discretization of the spatial domain

n = 40

x0 = -1
x1 = 1


y0 = -1
y1 = 1



xi = np.linspace(x0,x1,n)
yi = np.linspace(y0,y1,n)

Xi, Yi = np.meshgrid(xi,yi)

h = xi[1]-xi[0]




#Discretization of the temporal domain

t0 = 0
t1 = 1
nt = 2500

ti = np.linspace(t0,t1,nt)
dt = ti[1]-ti[0]




#Generating the matrix with initial values

U = np.zeros((n,n,nt))

U[:,:,0] = CI2D(Xi,Yi)

U = U.reshape((n**2,nt))




#Defining the matrices that discretize the Laplacian

diag = -4*np.ones(n**2)
o1 = np.ones(n**2-1)
o2 = np.ones(n**2-n)


A = sparse.diags([diag, o1, o1, o2, o2], [0,1,-1,n,-n],shape=(n**2,n**2))


A1 = (sparse.eye(n**2)-(D*dt)/(2*h**2)*A).toarray()
A2 = (sparse.eye(n**2)+(D*dt)/(2*h**2)*A).toarray()




#Imposing boundary conditions on matrices A1, A2

for i in range(n):
    A2[i,:] = np.zeros(n**2)
    A2[i*n,:] = np.zeros(n**2)
    A2[i*n + n-1,:] = np.zeros(n**2)
    A2[-i-1,:] = np.zeros(n**2)

    A1[i,:] = np.zeros(n**2)
    A1[i*n,:] = np.zeros(n**2)
    A1[i*n + n-1,:] = np.zeros(n**2)
    A1[-i-1,:] = np.zeros(n**2)

    A1[i,i] = 1
    A1[i*n,i*n] = 1
    A1[-i-1,-i-1] = 1
    
    A1[i*n + n-1,i*n + n-1] = 1
    A1[i*n + n-1,i*n + n-2] = -1




#Loop for time propagation of the differential equation

B = sparse.csc_matrix(np.dot(np.linalg.inv(A1),A2))

for j in range(nt-1):
    U[:,j+1] = B.dot(U[:,j])

U = U.reshape((n,n,nt))


animacion2D(U,Xi,Yi,ti,15,'Time evolution of temperature in 2D')




'''
Part 3
Advection-Diffusion Equation
Equation: $\frac{\partial u}{\partial t} = D \nabla^2 u - \mathbf{v} \cdot \nabla u$
'''

def vx(x,y):
    '''
    X-component of velocity
    '''
    return np.sin(np.pi*x)*np.cos(np.pi*y)

def vy(x,y):
    '''
    Y-component of velocity
    '''
    return -np.cos(np.pi*x)*np.sin(np.pi*y)
    
#Defining the constants needed

D = 0.1



#Discretization of the spatial domain

n = 40

x0 = -1
x1 = 1


y0 = -1
y1 = 1



xi = np.linspace(x0,x1,n)
yi = np.linspace(y0,y1,n)

Xi, Yi = np.meshgrid(xi,yi)

h = xi[1]-xi[0]




#Discretization of the temporal domain

t0 = 0
t1 = 1
nt = 2500

ti = np.linspace(t0,t1,nt)
dt = ti[1]-ti[0]




#Generating the matrix with initial values

U = np.zeros((n,n,nt))

U[:,:,0] = CI2D(Xi,Yi)

U = U.reshape((n**2,nt))




#Defining the matrices that discretize the Laplacian

diag = -4*np.ones(n**2)
o1 = np.ones(n**2-1)
o2 = np.ones(n**2-n)


A = sparse.diags([diag, o1, o1, o2, o2], [0,1,-1,n,-n],shape=(n**2,n**2))



#Defining the matrix that discretizes the Advection term

vx1 = vx(Xi,Yi).reshape(n**2)[0:-1]
vx2 = -vx(Xi,Yi).reshape(n**2)[1:]

vy1 = vy(Xi,Yi).reshape(n**2)[0:-n]
vy2 = -vy(Xi,Yi).reshape(n**2)[n:]


B = sparse.diags([vx1, vx2, vy1, vy2], [1,-1,n,-n],shape=(n**2,n**2))


A1 = (sparse.eye(n**2)-(D*dt)/(2*h**2)*A-(dt)/(4*h)*B).toarray()
A2 = (sparse.eye(n**2)+(D*dt)/(2*h**2)*A+(dt)/(4*h)*B).toarray()




#Imposing boundary conditions on matrices A1, A2

for i in range(n):
    A2[i,:] = np.zeros(n**2)
    A2[i*n,:] = np.zeros(n**2)
    A2[i*n + n-1,:] = np.zeros(n**2)
    A2[-i-1,:] = np.zeros(n**2)

    A1[i,:] = np.zeros(n**2)
    A1[i*n,:] = np.zeros(n**2)
    A1[i*n + n-1,:] = np.zeros(n**2)
    A1[-i-1,:] = np.zeros(n**2)

    A1[i,i] = 1
    A1[i*n,i*n] = 1
    A1[-i-1,-i-1] = 1
    A1[i*n + n-1,i*n + n-1] = 1




#Loop for time propagation of the differential equation

B = sparse.csc_matrix(np.dot(np.linalg.inv(A1),A2))

for j in range(nt-1):
    U[:,j+1] = B.dot(U[:,j])

U = U.reshape((n,n,nt))


animacion2D(U,Xi,Yi,ti,15,'Time evolution of temperature in 2D with Advection')