import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Solving EDOs
Marco Mas Sempere
'''

def plot(x, y, titulo, xlab, ylab, ylog = False, xlog = False, square = False, Label = False) -> None:
    '''
    Function that plots y vs x with various formatting options
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


G = 6.6738*10**(-11)
M = 1.9891*10**(30)
m = 5.9722*10**(24)


def Energia(x,y,vx,vy) -> np.ndarray:
    '''
    Function that calculates Energies (Potential U, Kinetic T, Total E)
    $U = -G \frac{Mm}{r}$
    $T = \frac{1}{2} m v^2$
    '''

    U = -G*M*m/((x**2 + y**2)**(1/2))
    T = m*(vx**2 + vy**2)/2

    return U,T,U+T




def dvdt(r) -> np.ndarray:
    '''
    Calculates acceleration (dv/dt)
    '''
    x, y = r
    return np.array([-G*M*x/((x**2 + y**2)**(3/2)), -G*M*y/((x**2 + y**2)**(3/2))])



def drdt(r) -> np.ndarray:
    '''
    Calculates the derivative of the state vector (velocity and acceleration)
    '''
    x,y,vx,vy = r
    return np.array([vx,vy,-G*M*x/((x**2 + y**2)**(3/2)), -G*M*y/((x**2 + y**2)**(3/2))])


def drdtO(r,t) -> np.ndarray:
    '''
    Derivative function for ODEINT (includes time argument)
    '''
    x,y,vx,vy = r
    return np.array([vx,vy,-G*M*x/((x**2 + y**2)**(3/2)), -G*M*y/((x**2 + y**2)**(3/2))])



r0 = np.array([1.4719*10**(11),0,0,3.0287*10**(4)])
t1 = 157680000
t0 = 0
dt = 3600
t = np.arange(t0,t1 + dt,dt)
N = len(t)



def Verlet(r0,dvdt,dt,N) -> np.ndarray:
    '''
    Function that implements the Velocity Verlet method
    $r(t+\Delta t) = r(t) + v(t)\Delta t + \frac{1}{2}a(t)\Delta t^2$
    $v(t+\Delta t) = v(t) + \frac{1}{2}(a(t) + a(t+\Delta t))\Delta t$
    '''
    r = np.zeros((2*N-1,4))

    r[0,:] = r0
    r[1,[2,3]] = r[0,(2,3)] + (dt/2)*dvdt(r[0,[0,1]])

    
    for i in range(2,2*N-2,2):
        r[i,[0,1]]   = r[i-2,[0,1]] + dt*r[i-1,[2,3]]
        k = dt*dvdt(r[i,[0,1]])
        r[i+1,[2,3]] = r[i-1,[2,3]] + k
        r[i,[2,3]]   = r[i-1,[2,3]] + k/2
    
    r[-1,[0,1]] = r[-3,[0,1]] + dt*r[-2,[2,3]]
    r[-1,[2,3]] = r[-2,[2,3]] + dt*dvdt(r[-1,[0,1]])/2

    Sol = r[::2,:]

    return Sol[:,0],Sol[:,1],Sol[:,2],Sol[:,3]



def Leapfrog(r0,drdt,dt,N) -> np.ndarray:
    '''
    Function that implements the Leapfrog method
    $r_{i+1} = r_{i-1} + 2 \Delta t f(t_i, r_i)$
    '''
    r = np.zeros((2*N-1,4))

    r[0,:] = r0
    r[1] = r[0] + (dt/2)*drdt(r[0])

    
    for i in range(2,2*N-2,2):
        r[i]   = r[i-2] + dt*drdt(r[i-1])
        r[i+1] = r[i-1] + dt*drdt(r[i])

    r[-1] = r[-3] + dt*drdt(r[-2])

    Sol = r[::2,:]

    return Sol[:,0],Sol[:,1],Sol[:,2],Sol[:,3]



def RK4(r0,drdt,dt,N) -> np.ndarray:
    '''
    Function that implements the Runge-Kutta 4 method
    Classical 4th order method
    '''
    r = np.zeros((N,4))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(r[i])
        k2 = dt*drdt(r[i] + k1/2)
        k3 = dt*drdt(r[i] + k2/2)
        k4 = dt*drdt(r[i] + k3)
        r[i+1] = r[i] + (k1+ 2*k2 + 2*k3 + k4)/6
    
    return r[:,0],r[:,1],r[:,2],r[:,3]



def Euler(r0,drdt,dt,N) -> np.ndarray:
    '''
    Function that implements the Euler method
    $y_{n+1} = y_n + h f(t_n, y_n)$
    '''
    r = np.zeros((N,4))

    r[0,:] = r0

    for i in range(N-1):
        r[i+1] = r[i] + dt*drdt(r[i])
    
    return r[:,0],r[:,1],r[:,2],r[:,3]






'''
Execution of all methods and comparison of results
'''



#VERLET

xV,yV,vxV,vyV = Verlet(r0,dvdt,dt,N) 
UV,TV,EV = Energia(xV,yV,vxV,vyV)


plot(t/31536000,(xV**2+yV**2)**(1/2)/(1.496*10**(11)),'Orbit Radius vs Time, Verlet','t(years)','r(AU)')
plot(yV/(1.496*10**(11)),xV/(1.496*10**(11)),'Trajectory, Verlet','x(AU)','y(AU)',square=True)

plot(t/31536000,UV, 'Energies, Verlet','t(years)','E(J)',Label='U')
plot(t/31536000,TV, 'Energies, Verlet','t(years)','E(J)',Label='T')
plot(t/31536000,EV, 'Energies, Verlet','t(years)','E(J)',Label='E')

plt.show()





#LEAPFROG

xL,yL,vxL,vyL = Leapfrog(r0,drdt,dt,N) 
UL,TL,EL = Energia(xL,yL,vxL,vyL)


plot(t/31536000,(xL**2+yL**2)**(1/2)/(1.496*10**(11)),'Orbit Radius vs Time, Leapfrog','t(years)','r(AU)')
plot(yL/(1.496*10**(11)),xL/(1.496*10**(11)),'Trajectory, Leapfrog','x(AU)','y(AU)',square=True)

plot(t/31536000,UL, 'Energies, Leapfrog','t(years)','E(J)',Label='U')
plot(t/31536000,TL, 'Energies, Leapfrog','t(years)','E(J)',Label='T')
plot(t/31536000,EL, 'Energies, Leapfrog','t(years)','E(J)',Label='E')

plt.show()




#RUNGE KUTTA 4

xRK,yRK,vxRK,vyRK = RK4(r0,drdt,dt,N) 
URK,TRK,ERK = Energia(xRK,yRK,vxRK,vyRK)



plot(t/31536000,(xRK**2+yRK**2)**(1/2)/(1.496*10**(11)),'Orbit Radius vs Time, Runge-Kutta 4','t(years)','r(AU)')
plot(yRK/(1.496*10**(11)),xRK/(1.496*10**(11)),'Trajectory, Runge-Kutta 4','x(AU)','y(AU)',square=True)

plot(t/31536000,URK, 'Energies, Runge-Kutta 4','t(years)','E(J)',Label='U')
plot(t/31536000,TRK, 'Energies, Runge-Kutta 4','t(years)','E(J)',Label='T')
plot(t/31536000,ERK, 'Energies, Runge-Kutta 4','t(years)','E(J)',Label='E')

plt.show()





#EULER

xE,yE,vxE,vyE = Euler(r0,drdt,dt,N) 
UE,TE,EE = Energia(xE,yE,vxE,vyE)



plot(t/31536000,(xE**2+yE**2)**(1/2)/(1.496*10**(11)),'Orbit Radius vs Time, Euler','t(years)','r(AU)')
plot(yE/(1.496*10**(11)),xE/(1.496*10**(11)),'Trajectory, Euler','x(AU)','y(AU)',square=True)

plot(t/31536000,UE, 'Energies, Euler','t(years)','E(J)',Label='U')
plot(t/31536000,TE, 'Energies, Euler','t(years)','E(J)',Label='T')
plot(t/31536000,EE, 'Energies, Euler','t(years)','E(J)',Label='E')

plt.show()




#ODEINT

r = odeint(drdtO,r0,t)


xO,yO,vxO,vyO = r[:,0],r[:,1],r[:,2],r[:,3]
UO,TO,EO = Energia(xO,yO,vxO,vyO)


plot(t/31536000,(xO**2+yO**2)**(1/2)/(1.496*10**(11)),'Orbit Radius vs Time, ODEINT','t(years)','r(AU)')
plot(yO/(1.496*10**(11)),xO/(1.496*10**(11)),'Trajectory, ODEINT','x(AU)','y(AU)',square=True)

plot(t/31536000,UO, 'Energies, ODEINT','t(years)','E(J)',Label='U')
plot(t/31536000,TO, 'Energies, ODEINT','t(years)','E(J)',Label='T')
plot(t/31536000,EO, 'Energies, ODEINT','t(years)','E(J)',Label='E')

plt.show()



#METHOD COMPARISON

plot(t/31536000,(xO**2+yO**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius','t(years)','r(AU)',Label = 'ODEINT')
plot(t/31536000,(xV**2+yV**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius','t(years)','r(AU)',Label = 'Verlet')
plot(t/31536000,(xL**2+yL**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius','t(years)','r(AU)',Label = 'Leapfrog')
plot(t/31536000,(xRK**2+yRK**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius','t(years)','r(AU)',Label = 'Runge-Kutta 4')
plot(t/31536000,(xE**2+yE**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius','t(years)','r(AU)',Label = 'Euler')

plot(t/31536000,EO, 'Comparison of Total Energies','t(years)','E(J)',Label='ODEINT')
plot(t/31536000,EV, 'Comparison of Total Energies','t(years)','E(J)',Label='Verlet')
plot(t/31536000,EL, 'Comparison of Total Energies','t(years)','E(J)',Label='Leapfrog')
plot(t/31536000,ERK, 'Comparison of Total Energies','t(years)','E(J)',Label='Runge-Kutta 4')
plot(t/31536000,EE, 'Comparison of Total Energies','t(years)','E(J)',Label='Euler')

plt.show()


'''
If we zoom in, we observe that except for Euler's method, all others overlap. That is,
Verlet, Leapfrog, ODEINT, and Runge-Kutta 4 obtain good and fairly conservative results. 
However, Euler's method yields poorly conservative and generally poor results.
'''



#COMPARISON WITHOUT EULER

'''
Since Euler's method yields results very different from the other 3, I repeat the plots excluding Euler.
'''

plot(t/31536000,(xO**2+yO**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius (No Euler)','t(years)','r(AU)',Label = 'ODEINT')
plot(t/31536000,(xV**2+yV**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius (No Euler)','t(years)','r(AU)',Label = 'Verlet')
plot(t/31536000,(xL**2+yL**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius (No Euler)','t(years)','r(AU)',Label = 'Leapfrog')
plot(t/31536000,(xRK**2+yRK**2)**(1/2)/(1.496*10**(11)),'Comparison of Orbit Radius (No Euler)','t(years)','r(AU)',Label = 'Runge-Kutta 4')

plot(t/31536000,EO, 'Comparison of Total Energies (No Euler)','t(years)','E(J)',Label='ODEINT')
plot(t/31536000,EV, 'Comparison of Total Energies (No Euler)','t(years)','E(J)',Label='Verlet')
plot(t/31536000,EL, 'Comparison of Total Energies (No Euler)','t(years)','E(J)',Label='Leapfrog')
plot(t/31536000,ERK, 'Comparison of Total Energies (No Euler)','t(years)','E(J)',Label='Runge-Kutta 4')

plt.show()

'''
We observe that ODEINT is the least conservative and Runge-Kutta 4 the most. RK4 has such high precision
that even though the method is not symplectic (conservative by design), the deviations are negligible.
Regarding Leapfrog and Verlet, the latter conserves energy better, but in both cases
energy oscillates within an orbit, returning to the initial value when returning to the starting point.
However, deviations from constant energy are 7 orders of magnitude smaller than the energy itself.
In practical terms, they are all effectively conservative.
'''