import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from matplotlib import animation




'''
Schrödinger Equation
Marco Mas Sempere
'''


'''
Solving the Schrödinger Equation
Equation: $i \hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x)\psi$
Scaled units ($\hbar=1, m=1$): $\frac{\partial \psi}{\partial t} = \frac{i}{2} \frac{\partial^2 \psi}{\partial x^2} - i V(x)\psi$
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
    ax.set_ylabel('$|\psi(x,t)|^2$')
    line,= ax.plot(x ,F[:,0], label = 't = {} s'.format(0))
    L=ax.legend()
    
    def plotsim(i):
        line.set_data(x,F[:,i*k])
        L.get_texts()[0].set_text('t = {} s'.format(round(i*k*dt,2)))
        
        return line
    
    ani = animation.FuncAnimation(fig,plotsim,frames = Nt//k,interval = 20)
    plt.show()



#Defining an integral function

def integral(f,Dx) -> np.ndarray:
    return np.sum(f*Dx)


def Xesperado(f,X,Dx) -> np.ndarray:
    '''
    Calculates the expectation value of position <x>
    '''
    XESP = []

    for i in range(len(f[0,:])):
        XESP.append(integral(f[:,i]*X,Dx))
    
    return np.array(XESP)


def CI1(x) -> np.ndarray:
    '''
    Function that returns the initial conditions as a Gaussian packet
    '''
    return 1/(0.175**(1/2))*np.exp(-50*(x-0.5)**2)


def CI2(x) -> np.ndarray:
    '''
    Function that returns the initial conditions as a Gaussian packet with initial momentum
    '''
    return 1/(0.175**(1/2))*np.exp(-50*(x-0.5)**2)*np.exp(-10j*x)


def CI3(x) -> np.ndarray:
    '''
    Function that returns the initial conditions as a Gaussian packet with different parameters
    ''' 

    return 1/((0.175*0.583)**(1/2))*np.exp(-150*(x-0.5)**2)*np.exp(-3j*x)

 
def fV1(x) -> np.ndarray:
    '''
    Function that calculates the Harmonic Potential
    $V(x) = k(x-x_0)^2$
    '''

    k = 10000

    return k*(x-0.5)**2


def fV2(x) -> np.ndarray:
    '''
    Function that calculates the Infinite Well Potential
    (Zero inside, boundaries handled by grid)
    '''


    return np.zeros(len(x))




'''
Schrödinger Solver using Crank-Nicolson
Method: $(I - \frac{i \Delta t}{2} H) \psi^{n+1} = (I + \frac{i \Delta t}{2} H) \psi^n$
'''


def Schrodinger(V,CI,x,t):



    Nx = len(x)
    Nt = len(t)

    Dx = x[1]-x[0]
    Dt = t[1]-t[0]

    #Defining a constant needed (representing i/2 in the diffusion term)
    D = 0.5j

    #Generating the diagonals for the matrix discretizing the second derivative

    diag = -2*np.ones(Nx,dtype=complex)
    over = np.ones(Nx-1,dtype=complex)
    under = np.ones(Nx-1,dtype=complex)


    diag[0],diag[-1],over[0],under[-1] = 0,0,0,0

    A = sparse.diags([diag, over, under], [0,1,-1],shape=(Nx,Nx),dtype=complex).toarray()

    Mv = sparse.diags([V(x)], [0],shape=(Nx,Nx),dtype=complex)

    I = sparse.diags([diag/(-2)], [0],shape=(Nx,Nx),dtype=complex)

    #Implicit step matrix (LHS)
    A1 = -(Dt*D)/(2*Dx**2)*A + sparse.eye(Nx) + 0.5j*Mv*Dt

    #Explicit step matrix (RHS)
    A2 = (Dt*D)/(2*Dx**2)*A + I - 0.5j*Mv*Dt

    #Imposing Boundary Conditions (Dirichlet: psi=0 at edges)
    A1[0],A1[-1] = np.zeros(Nx),np.zeros(Nx)
    A1[0,0],A1[-1,-1] = 1,1

    A2[0],A2[-1] = np.zeros(Nx),np.zeros(Nx)



    B = sparse.csc_matrix(np.dot(np.linalg.inv(A1),A2))



    #Generating the matrix with initial values

    U = np.zeros((Nx,Nt),dtype=complex)

    U[:,0] = CI(x)

    P = []

    #Loop for time propagation


    for j in range(Nt-1):
        P.append(integral(np.abs(U[:,j])**2,Dx))
        U[:,j+1] = B.dot(U[:,j])

    P.append(integral(np.abs(U[:,Nt-1])**2,Dx))

    P = np.array(P)

    return U,P




'''
Scenario 1: Harmonic Oscillator
'''




#Discretization of the spatial domain


x0 = 0
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




#Solving for the desired case


U, P  = Schrodinger(fV1,CI1,xi,ti)



animacion1D(np.abs(U)**2,xi,ti,5,'Wave function evolution for Harmonic Potential')

plt.figure()
plt.title('Total probability vs time for Harmonic Potential')
plt.plot(ti,P)
plt.xlabel('t')
plt.ylabel('P')
plt.ylim((0,1.1*np.max(P)))


plt.figure()
plt.title('Expectation value of position for Harmonic Potential')
plt.plot(ti,Xesperado(np.abs(U)**2,xi,dx))
plt.ylim((x0,x1))
plt.xlabel('t')
plt.ylabel('<x>')
plt.show()




'''
Scenario 2: Infinite Potential Well
'''





#Discretization of the spatial domain


x0 = 0
x1 = 1
nx = 800

xi = np.linspace(x0,x1 ,nx)
dx = xi[1]-xi[0]



#Discretization of the temporal domain

t0 = 0
t1 = 1
nt = 10000

ti = np.linspace(t0,t1,nt)
dt = ti[1]-ti[0]



#Solving for the desired case


U, P  = Schrodinger(fV2,CI1,xi,ti)



animacion1D(np.abs(U)**2,xi,ti,5,'Wave function evolution for Infinite Well')


plt.figure()
plt.title('Total probability vs time for Infinite Well')
plt.plot(ti,P)
plt.xlabel('t')
plt.ylabel('P')
plt.ylim((0,1.1*np.max(P)))


plt.figure()
plt.title('Expectation value of position for Infinite Well')
plt.plot(ti,Xesperado(np.abs(U)**2,xi,dx))
plt.ylim((x0,x1))
plt.xlabel('t')
plt.ylabel('<x>')
plt.show()



'''
Scenario 3: Harmonic Potential with Initial Momentum
'''





#Discretization of the spatial domain

x0 = 0
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





U, P  = Schrodinger(fV1,CI2,xi,ti)



animacion1D(np.abs(U)**2,xi,ti,3,'Wave function evolution for Harmonic Potential with Initial Momentum')


plt.figure()
plt.title('Expectation value of position (Harmonic + Momentum)')
plt.plot(ti,Xesperado(np.abs(U)**2,xi,dx))
plt.ylim((0,1))
plt.xlabel('t')
plt.ylabel('<x>')



plt.figure()
plt.title('Total probability vs time (Harmonic + Momentum)')
plt.plot(ti,P)
plt.xlabel('t')
plt.ylabel('P')
plt.ylim((0,1.1*np.max(P)))
plt.show()