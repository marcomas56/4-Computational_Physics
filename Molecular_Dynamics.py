import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy.linalg as la
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

'''
Block 2 - Practice 4
Molecular Dynamics: Crystal Lattice & Time Evolution
Marco Mas Sempere
'''


'''
Lattice Definitions and Helper Functions
'''

# Basis vectors for different crystal structures
cubic    = np.array([[0,0,0.]])
bcc      = np.array([[0,0,0],[0.5,0.5,0.5]])
fcc      = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
diamond  = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],
                     [0.25,0.25,0.25],[0.75,0.75,0.25],[0.75,0.25,0.75],[0.25,0.75,0.75]])


def plot(x, y, titulo, xlab, ylab, ylog = False, xlog = False, square = False, Label = False, YLIM = False) -> None:
    '''
    Generic plotting function
    '''
    plt.figure(titulo)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if ylog:
        plt.semilogy()
    if xlog:
        plt.semilogx()
    if Label != False:
        plt.plot(x,y,label = Label)
        plt.legend()
    if not Label:
        plt.plot(x,y)
    if square:
        plt.axis('square')
    if YLIM:
        plt.ylim(0.9*np.min(y),1.1*np.max(y))



def grafica3D(x,y,z,energias = False,titulo = False) -> None:
    '''
    Plots a set of atoms in 3D. 
    If 'energias' is provided, color-codes atoms by energy/force magnitude.
    '''
    if type(energias) == np.ndarray:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter3D(x, y, z,c = energias,cmap = 'inferno')
        cbar = fig.colorbar(p, ax= ax, label = '$|F|(\mathring{A}kg/s^2)$')
        p.set_clim(1.1*np.min(energias),0.9*np.max(energias))
        if type(titulo) == bool:
            ax.set_title('Energies (eV) of nuclei in FCC lattice')
        if type(titulo) == str:
            ax.set_title(titulo)
        ax.set_xlabel('$x(\mathring{A})$')
        ax.set_ylabel('$y(\mathring{A})$')
        ax.set_zlabel('$z(\mathring{A})$')


    if type(energias) == bool:
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z)
        ax.set_title('Lattice Structure')
        ax.set_xlabel('$x(\mathring{A})$')
        ax.set_ylabel('$y(\mathring{A})$')
        ax.set_zlabel('$z(\mathring{A})$')



def grafica3DFuerza(x,y,z,fx,fy,fz,modulo,titulo = False) -> None:
    '''
    Plots the atoms and the force vectors acting on them
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter3D(x, y, z,c = modulo,cmap = 'inferno')
    # Quiver plot for force vectors
    q = ax.quiver(x, y, z, fx, fy, fz, color = 'red', length=1.5, normalize = True)
    cbar = fig.colorbar(p, ax= ax, label = '$|F|(\mathring{A}kg/s^2)$')
    p.set_clim(1.1*np.min(modulo),0.9*np.max(modulo))
    if type(titulo) == bool:
        ax.set_title('Forces on nuclei')
    if type(titulo) == str:
        ax.set_title(titulo)
    ax.set_xlabel('$x(\mathring{A})$')
    ax.set_ylabel('$y(\mathring{A})$')
    ax.set_zlabel('$z(\mathring{A})$')


def animacion(REDT,t,k) -> None:
    '''
    Function that animates the lattice evolution over time
    '''

    Nt = len(t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(REDT[0,:,0], REDT[0,:,1], REDT[0,:,2], label = 't = 0 fs')
    time_label = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    ax.set_title('Lattice Structure Evolution')
    ax.set_xlim(np.min(REDT[:,:,0]),np.max(REDT[:,:,0]))
    ax.set_ylim(np.min(REDT[:,:,1]),np.max(REDT[:,:,1]))
    ax.set_zlim(np.min(REDT[:,:,2]),np.max(REDT[:,:,2]))
    ax.set_xlabel('$x(\mathring{A})$')
    ax.set_ylabel('$y(\mathring{A})$')
    ax.set_zlabel('$z(\mathring{A})$')
    ax.set_aspect('equal')


    def plotsim(i):
        scatter._offsets3d = (REDT[i*k,:,0], REDT[i*k,:,1], REDT[i*k,:,2])
        time_label.set_text("t = {} fs".format(round(t[i*k]*10**(15))))
        
        return scatter,
    
    ani = animation.FuncAnimation(fig,plotsim,frames = Nt//k,interval = 20)
    plt.show()


def trayectoria(REDT,t) -> None:
    '''
    Plots the trajectory of a specific particle (central one)
    '''
    Nt = len(t)
    N = len(REDT[0,:,0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plotting displacement relative to initial position
    ax.plot(REDT[:,N//2,0]-REDT[0,N//2,0], REDT[:,N//2,1]-REDT[0,N//2,1], REDT[:,N//2,2]-REDT[0,N//2,2])
    ax.set_title('Single Particle Trajectory')
    ax.set_xlabel('$x-x_0(\mathring{A})$')
    ax.set_ylabel('$y-y_0(\mathring{A})$')
    ax.set_zlabel('$z-z_0(\mathring{A})$')
    ax.set_aspect('equal')




def red3D(nx,ny,nz,base) -> np.ndarray:
    '''
    Function that generates the lattice given dimensions and basis.
    Vectorized implementation (Meshgrid + Broadcasting) for efficiency.
    '''

    centro = np.array([nx-1,ny-1,nz-1])*0.5
    i,j,k = np.meshgrid(np.arange(nx),np.arange(ny),np.arange(nz), indexing='ij')
    pos_base=np.stack([i,j,k],axis = 3).reshape(nx*ny*nz,1,3)
    red = pos_base + base - centro

    return np.array(red.reshape(len(base)*nx*ny*nz,3))


'''
Physics Definitions
Lennard-Jones Potential: $V(r) = 4\varepsilon [(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^6]$
Force: $\vec{F} = -\nabla V(r)$
'''

sigma   = 2.3151                 # In Angstroms
epsilon = 0.167                  # In eV
a = 3.603                        # Lattice constant (Angstroms)

def V(r, s = sigma, E = epsilon) -> np.ndarray:
    '''
    Calculates Lennard-Jones Potential
    '''
    return 4*E*((s/r)**12-(s/r)**6)


def DistanciaRED(r0,RED) -> np.ndarray:
    '''Calculates distance from r0 to all points in RED'''
    return la.norm(r0-RED,axis = 1)


def Distancia(r0,r1) -> float:
    '''Calculates distance between two points'''
    return la.norm(r0-r1)


'''
User Input
'''

nx  = int(input('Number of cells in x-direction: '))
ny  = int(input('Number of cells in y-direction: '))
nz  = int(input('Number of cells in z-direction: '))


'''
Static Force Calculation
Representing the lattice colored by force magnitude and direction
'''

n = np.array([nx,ny,nz])
N = 4*nx*ny*nz

# Generate FCC Lattice
RED = a*red3D(nx,ny,nz,fcc)


def Fi(r) -> np.ndarray:       
    '''
    Calculates the force due to Lennard-Jones potential
    Analytical derivative of V(r)
    '''
    E = epsilon*1.6*10**(-19) # Convert eV to Joules
    s = sigma
    r = r
    x = la.norm(r,axis = 1)
    
    # F = -dV/dr * (r/|r|)
    A = -((24*E*(s**6)*(2*s**6-x[:,None]**6))/(x[:,None]**14))*r*(10**20)
    return A


def FTi(r,RED) -> np.ndarray:
    '''
    Calculates total force on a single particle from neighbors
    Cutoff: 3*sigma
    '''

    D = DistanciaRED(r,RED)
    # Filter neighbors within cutoff and exclude self (D>0)
    RED0 = RED[(D<3*sigma)*(D > 0)]
    F = Fi(RED0-r)

    return np.sum(F,axis = 0)


def FT(RED) -> np.ndarray:
    '''
    Calculates total force on EVERY particle in the lattice
    '''
    F = []
    for r in RED:
        F.append(FTi(r,RED))
    return np.array(F)


def MFT(RED) -> np.ndarray:
    '''Calculates magnitude of the total force vectors'''
    F = FT(RED).astype(float)
    F2 = F**2
    SF2 = np.sum(F2,axis = 1)
    return np.sqrt(SF2)

F = FT(RED)


# Plotting static forces
grafica3D(RED[:,0],RED[:,1],RED[:,2],MFT(RED),'Force exerted on each particle')
grafica3DFuerza(RED[:,0],RED[:,1],RED[:,2],F[:,0],F[:,1],F[:,2],MFT(RED),'Force exerted on each particle')


plt.show()

# EXTRA 1

'''
Velocity Distribution
Initialization of velocities based on Temperature.
Equipartition Theorem: $E_k = \frac{3}{2} k_B T$
'''

K = 8.6181024*10**(-5)                                 # Boltzmann constant eV/K
m = 63.55*1.66*10**(-27)                               # Mass (kg) (Copper)



def EC(V):                                              # Returns Kinetic Energy in eV
    V = V/(10**10) # Unit conversion
    return 0.5*m*np.sum(V**2,axis = (0,1))/(1.6*10**(-19))
    
def Temperatura(V):                                     # Returns Temperature in K
    return (2*EC(V))/(3*K*N)



Temp = float(input('Lattice Temperature (in K): '))


# Random initial velocities centered at 0
VA = np.random.rand(N,3)-0.5

# Scaling velocities to match desired Temperature
TR = Temperatura(VA)
sc = (Temp/TR)**(1/2)


VRED = VA*sc
TR = Temperatura(VRED)


print('Obtained Temperature: {} K'.format(round(TR,3)))



# EXTRA 2

'''
Molecular Dynamics Simulation
Time evolution using Verlet Algorithm.
'''

k = int(input('Enter number of steps: '))
dt = 10**(-15) # Time step: 1 femtosecond

t = np.arange(0,dt*(k),dt)


def Energías(RED,k = 3) -> np.ndarray:               # Potential Energy in eV

    E = np.zeros(len(RED[:,0]))
    i = 0
    for r0 in RED:
        Ei = 0
        for r1 in RED[DistanciaRED(r0,RED) < k*sigma]:
            d = Distancia(r0,r1)
            if d!= 0:
                Ei += V(d)
        E[i] = Ei
        i += 1


    return np.array(E)


def dvdt(RED) -> np.ndarray:
    '''Acceleration calculation a = F/m'''
    return FT(RED)/m


def VerletRED(RED0,VRED0,dvdt,dt,N,NRED) -> np.ndarray:
    '''
    Function that implements the Verlet Integration method
    '''
    r = np.zeros((N,NRED,3))
    v = np.zeros((N,NRED,3))
    v_bien = np.zeros((N,NRED,3))

    # Initial conditions
    r[0,:,:] = RED0
    v_bien[0,:,:] = VRED0
    v[0,:,:] = v_bien[0,:,:] + (dt/2)*dvdt(r[0,:,:])

    # Time stepping loop
    for i in range(1,N):
        r[i,:,:]   = r[i-1,:,:] + dt*v[i-1,:,:]
        k = dt*dvdt(r[i,:,:])
        v[i,:,:] = v[i-1,:,:] + k
        v_bien[i,:,:]   = v[i-1,:,:] + k/2
    
    # Correction for last step (commented out in original)
    #r[-1,:,:] = r[-3,:,:] + dt*v[-1,:,:]
    #v_bien[-1,:,:] = v[-3,:,:] + dt*dvdt(r[-1,:,:])/2

    return r,v_bien


# Running the simulation
RED_t, VRED_t = VerletRED(RED,VRED,dvdt,dt,k,N)




'''
Results Analysis
Plotting evolution of quantities of interest (T, U, Kinetic Energy, Total Energy)
Animation and Single Particle Trajectory
'''

T_t = []
Ec_t = []
U_t = []

U0 = np.sum(Energías(RED_t[0]))/2
T0 = Temperatura(VRED_t[0])
Ec0 = EC(VRED_t[0])


# Calculating energy at each time step
for i in range(k):
    T_t.append(Temperatura(VRED_t[i]))
    Ec_t.append(EC(VRED_t[i]))
    U_t.append(np.sum(Energías(RED_t[i]))/2)

T_t = np.array(T_t)
Ec_t = np.array(Ec_t)
U_t = np.array(U_t)

# Plotting fluctuations relative to initial state
plot(t*10**(15),T_t-T0,'Temperature vs Time','t(fs)','$T-T_0(K)$',Label = '$T_0 = {}K$'.format(round(T0,2)))
plot(t*10**(15),Ec_t-Ec0,'Kinetic Energy vs Time','t(fs)','$Ec-Ec_0(eV)$',Label = '$Ec_0 = {}eV$'.format(round(Ec0,2)))
plot(t*10**(15),U_t-U0,'Potential Energy vs Time','t(fs)','$U-U_0(eV)$',Label = '$U_0 = {}eV$'.format(round(U0,2)))
plot(t*10**(15),U_t+Ec_t,'Total Energy vs Time','t(fs)','$E(eV)$',YLIM = True)


plt.show()

trayectoria(RED_t,t)
animacion(RED_t,t,1)