import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from matplotlib import animation

'''
Advecrion and Burgues EDPs
Marco Mas Sempere
'''


def animacion1D(F,x,t,k,titulo) -> None:
    '''
    Function that animates a function F(t,x)
    '''
    Nt = len(t)
    dt = t[1]-t[0]

    fig = plt.figure()
    fig.suptitle(titulo)
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    line,= ax.plot(x ,F[0,:], label = 't = {} s'.format(0))
    L=ax.legend()
    
    def plotsim(i):
        line.set_data(x,F[i*k,:])
        L.get_texts()[0].set_text('t = {} s'.format(round(i*k*dt,1)))
        
        return line
    
    ani = animation.FuncAnimation(fig,plotsim,frames = Nt//k,interval = 20)
    plt.show()




#Part a)
#Advection Equation: $\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$


def u0(x):
    if 1 <= x <= 2:
        return 1
    else:
        return 0

u0 = np.vectorize(u0)


#Defining constants

c = 2





#Solving Advection using Central Differences
#Scheme: $u_i^{n+1} = u_i^n - \frac{c \Delta t}{2 \Delta x} (u_{i+1}^n - u_{i-1}^n)$

def CentradasAdv(x,t,c,U0) -> np.ndarray:
    '''
    Solves the advection equation using explicit method and central differences
    '''

    nt = len(t)
    nx = len(x)


    dx = x[1]-x[0]

    U = np.zeros((nt,nx))
    U[0,:] = U0(x)
    
    #Making the matrix that discretizes the method
    
    A = sparse.diags((-np.ones(nx-1),np.ones(nx-1)),(-1,1),(nx,nx)).toarray()
    I = np.eye(nx)

    B = I - (c*dt)/(2*dx)*A


    #Implementing boundary conditions


    B[0,:],B[-nx,:] = np.zeros(nx),np.zeros(nx)
    B[0,0],B[nx-1,nx-1] = 1,1 



    for j in range(nt-1):
        U[j+1,:] = np.dot(B,U[j,:])
        U[j+1,0] = U[j+1,-2]
        U[j+1,-1] = U[j+1,1]

    return U 


#Discretization of domains

nx = 50
x = np.linspace(0,5,nx)
dx = x[1]-x[0]


nt = 5000
t = np.linspace(0,6,nt)
dt = t[1]-t[0]



animacion1D(CentradasAdv(x,t,c,u0),x,t,10,'Advection using Central Differences')

'''
This method is unconditionally unstable for any initial conditions.
'''




#Solving Advection using Downwind Scheme
#Scheme: $u_i^{n+1} = u_i^n - \frac{c \Delta t}{\Delta x} (u_{i+1}^n - u_i^n)$


def DownwindAdv(x,t,c,u0) -> np.ndarray:
    '''
    Solves the advection equation using explicit method and downwind scheme
    '''

    nt = len(t)
    nx = len(x)

    dx = x[1]-x[0]

    U = np.zeros((nt,nx))
    U[0,:] = u0(x)
    
    #Making the matrix that discretizes the method
    
    A = sparse.diags((np.ones(nx-1)),(1),(nx,nx)).toarray()
    I = np.eye(nx)

    B = (1+c*dt/dx)*I - c*(dt)/(dx)*A


    #Implementing boundary conditions

    B[0,:],B[-nx,:] = np.zeros(nx),np.zeros(nx)
    B[0,0],B[nx-1,nx-1] = 1,1 

    for j in range(nt-1):
        U[j+1,:] = np.dot(B,U[j,:])
        U[j+1,0] = U[j+1,-2]
        U[j+1,-1] = U[j+1,1]

    return U 


#Discretization of domains

nx = 50
x = np.linspace(0,5,nx)
dx = x[1]-x[0]


nt = 10000
t = np.linspace(0,6,nt)
dt = t[1]-t[0]


animacion1D(DownwindAdv(x,t,c,u0),x,t,10,'Advection using Downwind Scheme')


'''
This method is also unstable for these initial conditions, as the propagation is in the 
positive direction of the x-axis (requires c < 0 for stability).
'''


#Solving Advection using Upwind Scheme
#Scheme: $u_i^{n+1} = u_i^n - \frac{c \Delta t}{\Delta x} (u_i^n - u_{i-1}^n)$

def UpwindAdv(x,t,c,u0) -> np.ndarray:
    '''
    Solves the advection equation using explicit method and upwind scheme
    '''

    nt = len(t)
    nx = len(x)

    dx = x[1]-x[0]

    U = np.zeros((nt,nx))
    U[0,:] = u0(x)
    
    #Making the matrix that discretizes the method
    
    A = sparse.diags((np.ones(nx-1)),(-1),(nx,nx)).toarray()
    I = np.eye(nx)

    B = (1-c*dt/dx)*I + c*(dt)/(dx)*A


    #Implementing boundary conditions

    B[0,:],B[-nx,:] = np.zeros(nx),np.zeros(nx)
    B[0,0],B[nx-1,nx-1] = 1,1
    

    for j in range(nt-1):
        U[j+1,:] = np.dot(B,U[j,:])
        U[j+1,0] = U[j+1,-2]
        U[j+1,-1] = U[j+1,1]

    return U 




#Discretization of domains

nx = 1000
x = np.linspace(0,5,nx)
dx = x[1]-x[0]


nt = 10000
t = np.linspace(0,6,nt)
dt = t[1]-t[0]


animacion1D(UpwindAdv(x,t,c,u0),x,t,10,'Advection using Upwind Scheme')

'''
This method is stable for these initial conditions, as the propagation is in the 
positive direction of the x-axis, provided the CFL condition is met.
'''





#Part b)
#Burgers' Equation (inviscid): $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0$


def u0(x):
    return 2+0.5 *np.sin(2*np.pi*x)




#Solving Burgers' Equation using Central Differences

def CentradasBur(x,t,U0) -> np.ndarray:
    '''
    Solves Burgers' equation using explicit method and central differences
    '''

    nt = len(t)
    nx = len(x)

    dt = t[1]-t[0]
    dx = x[1]-x[0]

    U = np.zeros((nt,nx))
    U[0,:] = U0(x)
    


    
    #Making the matrix that discretizes the method
    
    A = -(dt/(2*dx))*(sparse.diags((np.ones(nx-1),-np.ones(nx-1)),(1,-1),(nx,nx)).toarray())



    #Implementing boundary conditions


    A[0,:],A[-1,:] = np.zeros(nx),np.zeros(nx)



    for j in range(nt-1):
        U[j+1,:] = U[j,:]*(np.ones(nx) + np.dot(A,U[j,:]))
        U[j+1,0] = U[j+1,-2]
        U[j+1,-1] = U[j+1,1]


    return U 


#Discretization of domains

nxC = 50
xC = np.linspace(0,1,nxC)
dxC = xC[1]-xC[0]


ntC = 10000
tC = np.linspace(0,2,ntC)
dtC = tC[1]-tC[0]

UCB = CentradasBur(xC,tC,u0)

animacion1D(UCB,xC,tC,10,'Burgers Equation using Central Differences')


'''
This method is poorly stable for these initial conditions. Stability depends on 
local velocity u(x,t) and meeting CFL condition.
'''


#Solving Burgers' Equation using Upwind Scheme
#Conservative form: $\frac{\partial u}{\partial t} + \frac{1}{2}\frac{\partial (u^2)}{\partial x} = 0$

def UpwindBur(x,t,u0) -> np.ndarray:
    '''
    Solves Burgers' equation using explicit method and upwind scheme
    '''

    nt = len(t)
    nx = len(x)

    dx = x[1]-x[0]

    U = np.zeros((nt,nx))
    U[0,:] = u0(x)
    
    #Making the matrix that discretizes the method
    
    A = -(dt/(2*dx))*sparse.diags((np.ones(nx),-np.ones(nx-1)),(0,-1),(nx,nx)).toarray()
    I = np.eye(nx)

   


    #Implementing boundary conditions

    A[0,:],A[-1,:] = np.zeros(nx),np.zeros(nx)

    
    for j in range(nt-1):
        U[j+1,:] = np.dot(I,U[j,:])+np.dot(A,U[j,:]**2)
        U[j+1,0] = U[j+1,-2]
        U[j+1,-1] = U[j+1,1]

    return U 


#Discretization of domains

nxU = 500
xU = np.linspace(0,1,nxU)
dxU = xU[1]-xU[0]


ntU = 10000
tU = np.linspace(0,2,ntU)
dtU = tU[1]-tU[0]

UUB = UpwindBur(xU,tU,u0)

animacion1D(UUB,xU,tU,5,'Burgers Equation using Upwind Scheme')

'''
This method is stable for these initial conditions, as the propagation is in the 
positive direction of the x-axis, provided the CFL condition is met.
'''


#Comparison of Energies
#Energy: $E(t) = \frac{1}{2} \int u^2 dx$

ECen = np.zeros(ntC)
for i in range(ntC):
    ECen[i] = 0.5*np.sum((UCB[i,:]**2)*dxC)

EUp = np.zeros(ntU)
for i in range(ntU):
    EUp[i] = 0.5*np.sum((UUB[i,:]**2)*dxU)


plt.figure('Comparison of Energies')
plt.title('Comparison of Energies')
plt.plot(tC,ECen, label = 'Central Differences')
plt.plot(tU,EUp, label = 'Upwind')
#plt.semilogy()
plt.legend()
plt.xlabel('t')
plt.ylabel('E')
plt.show()




#Extra
#Solving with mixed Upwind/Downwind based on velocity sign


def u0(x):
    return np.sin(2*np.pi*x) 

def Extra(x,t,u0) -> np.ndarray:
    '''
    Solves the advection equation using a mixture of upwind and downwind
    '''

    nt = len(t)
    nx = len(x)

    dx = x[1]-x[0]

    U = np.zeros((nt,nx))
    U[0,:] = u0(x)
    

    
    
    for j in range(nt-1):
        for n in range(1,nx-1):
            if U[j,n] > 0:
                U[j+1,n] = U[j,n]- (dt/(2*dx))*((U[j,n])**2-(U[j,n-1])**2)
            else:
                U[j+1,n] = U[j,n]- (dt/(2*dx))*((U[j,n+1])**2-(U[j,n])**2)

        U[j+1,0] = 0
        U[j+1,-1] = 0

    return U 
#Discretization of domains

nx = 500
x = np.linspace(0,1,nx)
dx = x[1]-x[0]


nt = 5000
t = np.linspace(0,2,nt)
dt = t[1]-t[0]


Ext = Extra(x,t,u0)

animacion1D(Ext,x,t,15,'Extra: Burgers Equation with u0(x) = sin(2pix)')


EExt = np.zeros(nt)
for i in range(nt):
    EExt[i] = 0.5*np.sum((Ext[i,:]**2)*dx)


plt.figure('Energy vs Time (Extra)')
plt.title('Energy vs Time for the Extra case')
plt.plot(t,EExt)
plt.xlabel('t')
plt.ylabel('E')
plt.show()