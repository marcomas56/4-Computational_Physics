import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

'''
Monte Carlo Simulation: Canonical Ensemble
Metropolis Algorithm for Particles in a 3D Box
Marco Mas Sempere
'''


'''
Initial Setup
We request user parameters, initialize arrays and constants.
System: Ideal gas of N particles in a 3D Infinite Potential Well.
Energy levels: $E_{\vec{n}} = \frac{\pi^2 \hbar^2}{2mL^2} (n_x^2 + n_y^2 + n_z^2)$
(Using natural units where constants = 1)
'''

N = int(input('Number of particles (N): '))
KT = float(input('Thermal Energy (KT): '))
k = int(input('Number of Monte Carlo steps (k): '))



h = 1
L = 1
m = 1

'''
Metropolis Algorithm Implementation
Evolution of the system towards equilibrium.
'''


def Energia(n):
    '''
    Calculates the total energy of the system based on quantum numbers n
    '''
    return (np.pi**2)*(np.linalg.norm(n)**2)/2

def EvoMetropolis(N,k,KT):
    '''
    Evolves the system using the Metropolis Algorithm.
    1. Propose a change in state (dn = +1 or -1).
    2. Calculate energy difference dE.
    3. Accept or reject based on Probability $P = \min(1, e^{-\Delta E / k_B T})$.
    '''

    # Initial state: Ground state (n=1 for all dimensions)
    n0 = np.ones((N,3))
    E = Energia(n0)

    Ei = [E]
    
    # Pre-calculate random indices for efficiency
    nR = rand.randint(np.array([0,0]),np.array([N,3]),(k,2))

    for i in range(k):

        nRi = nR[i]
        ni = n0[nRi[0],nRi[1]]     # Selected quantum number
        dn = rand.choice([-1,1])   # Proposed change

        # Boundary condition: n must be > 0 (Quantum Box)
        if dn + ni > 0:

            a = rand.uniform(0,1)

            # Calculate Energy Difference
            # Delta E proportional to: (n+dn)^2 - n^2 = 2*n*dn + 1 (since dn^2=1)
            dE =  (np.pi**2)*(2*dn*ni + 1)/2
            
            # Metropolis Criterion
            # If dE < 0, exp > 1, P is True (Accept).
            # If dE > 0, check against random number 'a'.
            P = np.e**(-dE/KT) > a
            
            # Update state if Accepted (P is True/1)
            n0[nRi[0],nRi[1]] = ni + dn*P

            # Update Energy tracking (if rejected, P=0, dE=0)
            dE = dE*P

        else:
            dE =  0 # Move rejected due to boundary
        
        Ei.append(Ei[i] + dE)
        
    return np.array(Ei)


# Running the simulation
Ef = EvoMetropolis(N,k,KT)

# Plotting the result
plt.figure()
plt.title('Total Energy vs Time Steps')
plt.plot(Ef)
plt.ylabel('Energy (E)')
plt.xlabel('Monte Carlo Steps')
plt.show()




'''
Analysis 1: Varying Temperature (KT)
Observing how thermal energy affects the equilibrium state.
'''


KTs = np.logspace(0.5,3,4)
N = 100
k = 10000


plt.figure()
plt.title('Energy evolution for different Temperatures (KT)')
plt.semilogy()

for KT in KTs:
    Ef = EvoMetropolis(N,k,KT)
    plt.plot(Ef,label = 'KT = {}'.format(round(KT,1)))

plt.ylabel('Energy (E)')
plt.xlabel('Monte Carlo Steps')
plt.legend()
plt.show()


'''
Analysis:
We can observe that the KT factor changes the energy level at which the system 
reaches equilibrium. Since all executions use the same number of particles, 
this indicates that the average energy per particle increases with KT.
This is consistent with statistical mechanics, where the average energy 
is directly related to temperature.
'''




'''
Analysis 2: Varying Number of Particles (N)
Observing extensive property of Energy.
'''


Ns = np.round(np.logspace(1,3,4))
KT = 100
k = 10000


plt.figure()
plt.title('Energy evolution for different Particle Numbers (N)')
plt.semilogy()

for N in Ns:
    Ef = EvoMetropolis(int(N),k,KT)
    plt.plot(Ef,label = 'N = {}'.format(int(N)))

plt.ylabel('Energy (E)')
plt.xlabel('Monte Carlo Steps')
plt.legend()
plt.show()



'''
Analysis:
In this case, we see that the total energy increases, but this is primarily 
because we are increasing the number of particles (Energy is an extensive property).
'''