import numpy as np
import matplotlib.pyplot as plt


'''
Block 2 - Practice 1
Crystal Lattice Generation
Marco Mas Sempere
'''

# Requesting necessary data from the user
typ = input('Crystal lattice type (bcc, fcc, cubic, or diamond): ')
nx  = int(input('Number of cells in x-direction: '))
ny  = int(input('Number of cells in y-direction: '))
nz  = int(input('Number of cells in z-direction: '))



# Defining the basis vectors for each unit cell
# Basis vectors $\vec{r}_b$ in relative coordinates [0,1]

cubic    = np.array([[0,0,0.]])

bcc      = np.array([[0,0,0],
                     [0.5,0.5,0.5]])

fcc      = np.array([[0,0,0],
                     [0.5,0.5,0],
                     [0.5,0,0.5],
                     [0,0.5,0.5]])

diamond  = np.array([[0,0,0],
                     [0.5,0.5,0],
                     [0.5,0,0.5],
                     [0,0.5,0.5],
                     [0.25,0.25,0.25],
                     [0.75,0.75,0.25],
                     [0.75,0.25,0.75],
                     [0.25,0.75,0.75]])




def grafica3D(x,y,z) -> None:
    '''
    Plots the set of atoms in the lattice in 3D
    '''
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z)
    ax.set_title('Lattice Structure')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



def red3D1_0(nx,ny,nz,base) -> np.ndarray:
    '''
    Function that generates the lattice given the number of cells 
    in each direction and the basis element.
    Method: Explicit loops (Less efficient, more readable).
    Formula: $\vec{R} = \vec{R}_{cell} + \vec{r}_{basis} - \vec{R}_{center}$
    '''

    red = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # We subtract (N-1)/2 to center the lattice at the origin
                shift = np.array([(nx-1)/2, (ny-1)/2, (nz-1)/2], dtype=float)
                cell_pos = np.array([i,j,k], dtype=float)
                
                red.append((base + cell_pos - shift))
                
    return np.array(red).reshape(len(base)*nx*ny*nz,3)



def red3D2_0(nx,ny,nz,base) -> np.ndarray:
    '''
    Function that generates the lattice given the number of cells
    in each direction and the basis element.
    Method: Vectorized using meshgrid (Less readable, more efficient).
    '''

    centro = np.array([nx-1,ny-1,nz-1])*0.5
    
    # Generate grid of indices
    i,j,k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    
    # Stack and reshape to broadcast with basis
    pos_base = np.stack([i,j,k], axis=3).reshape(nx*ny*nz, 1, 3)
    
    # Apply broadcasting
    red = pos_base + base - centro

    return np.array(red.reshape(len(base)*nx*ny*nz, 3))



def Archivo(R,name) -> None:
    '''
    Function that writes the atom positions to a file
    Format:
    N_atoms
    Index: x, y, z
    '''
    archivo = open(name,'w')
    archivo.write(str(len(R[:,0])) +'\n')
    for i in range(len(R[:,0])):
        archivo.write(str(i+1) +': ' + str(R[i,0]) + ',' + str(R[i,1]) + ',' + str(R[i,2])+'\n')
    archivo.close()
    


# Determine which unit cell basis to use

if typ == 'cubic':
    base = cubic   
elif typ == 'diamond':
    base = diamond
elif typ == 'bcc':
    base = bcc
elif typ == 'fcc':
    base = fcc
else:
    print("Warning: Lattice type not recognized. Defaulting to cubic.")
    base = cubic


# Execution blocks
# We use the vectorized version (faster) by default. The loop version is commented out.

'''
red1 = red3D1_0(nx,ny,nz,base)
grafica3D(red1[:,0],red1[:,1],red1[:,2])
'''


red  = red3D2_0(nx,ny,nz,base)
#grafica3D(red[:,0],red[:,1],red[:,2])
Archivo(red,'red.txt')