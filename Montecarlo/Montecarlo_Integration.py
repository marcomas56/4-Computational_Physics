import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

'''
Monte Carlo Integration
Marco Mas Sempere
'''


'''
Function Definition and Parameters
We start by defining the function, requesting parameters from the user,
and defining the integration limits.
'''

def f(x):
    return (np.sin(1/(x*(2-x))))**2

x0 = 0
x1 = 2

M = float(input('Maximum value of the function (M): '))
N = int(input('Total number of steps (N): '))



'''
Monte Carlo Integration Function
Method: Rejection Sampling / Hit-or-Miss
Integral estimate: $I \approx \frac{N_{hits}}{N_{total}} \cdot A_{box}$
Error estimate: $\sigma \approx \sqrt{\frac{I(A-I)}{N}}$
'''


def IntMonteCarlo(f,x0,x1,N,M):
    A = (x1-x0)*M # Area of the bounding box

    # Generate N random points in the box [x0, x1] x [0, M]
    ri = rand.uniform(np.array([x0,0]),np.array([x1,M]),(N,2))

    # Count points below the curve f(x)
    NF = np.sum(f(ri[:,0]) > ri[:,1])

    I = NF*A/N

    s = np.sqrt(I*(A-I)/N)
    return I,s


def PrintMonteCarloValores(f,x0,x1,N,M,titulo):
    '''
    Visualizes the Monte Carlo integration by plotting the function
    and the accepted points (hits).
    '''
    xi = np.linspace(x0+0.001,x1-0.001,100000)

    ri = rand.uniform(np.array([x0,0]),np.array([x1,M]),(N,2))

    # Filter accepted points
    RF = ri[f(ri[:,0]) > ri[:,1]]

    plt.figure()
    plt.title(titulo)
    plt.plot(xi,f(xi))
    plt.plot(RF[:,0],RF[:,1],'.', label='Accepted Points')
    plt.ylabel('$f(x)$')
    plt.xlabel('x')
    plt.legend()
    plt.show()




'''
Execution and Plotting
For the function $f(x) = \sin^2(\frac{1}{x(2-x)})$
'''

x =  np.linspace(x0+0.01,x1-0.01,100000)


plt.figure()
plt.title('Function to integrate: $f(x) = \sin^2(\\frac{1}{x(2-x)})$')
plt.plot(x,f(x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

Int = IntMonteCarlo(f,x0,x1,N,M)
print('Integral value for f(x) = sin^2(1/(x(2-x))): {}, with error {}'.format(round(Int[0],10),round(Int[1],7)))


'''
Convergence Study
Analyzing how the integral value varies with the number of steps N,
fixing M = 1 (known analytical maximum).
'''


Ii = []
Ni = []
Si = []


for k in np.linspace(10,10000,1000):
    Int = IntMonteCarlo(f,x0,x1,round(k),1)
    Ii.append(Int[0])
    Si.append(Int[1])
    Ni.append(round(k))

Ii = np.array(Ii)
Ni = np.array(Ni)
Si = np.array(Si)

plt.figure()
plt.title('Integral value of $f(x) = \sin^2(\\frac{1}{x(2-x)})$ vs Number of steps')
plt.errorbar(Ni,Ii,yerr=Si, fmt = 'ro', ms=4, ecolor='r', elinewidth = 1,capsize = 4, capthick = 1)
plt.plot(Ni,Ii)
plt.ylabel('$\int fdx$')
plt.xlabel('Number of steps')
#plt.semilogx()
plt.show()


'''
Plotting Accepted Values
'''

PrintMonteCarloValores(f,x0,x1,N,M,'Monte Carlo Integration: Accepted Points for $f(x)$')


'''
Efficiency Analysis vs Box Height (M)
Analyzing how the result/error varies with the chosen maximum M.
Since the function has a maximum at 1, we test values from 1 to 1000 
with N = 100,000 fixed. Larger M means more wasted points (lower efficiency).
'''



Ii = []
xmi = []
Si = []


for M in np.logspace(0,3,100):
    Int = IntMonteCarlo(f,x0,x1,100000,M)
    Ii.append(Int[0])
    Si.append(Int[1])
    xmi.append(M)

Ii = np.array(Ii)
xmi = np.array(xmi)
Si = np.array(Si)

plt.figure()
plt.title('Integral value vs Maximum value M')
plt.errorbar(xmi,Ii,yerr=Si, fmt = 'ro', ms=4, ecolor='r', elinewidth = 1,capsize = 4, capthick = 1)
plt.plot(xmi,Ii)
plt.ylabel('$\int fdx$')
plt.xlabel('$M_{box}$')
plt.semilogx()
plt.show()



'''
Other Example
Gaussian Integral
Repeating the process for $g(x) = e^{-x^2}$ in interval [-5, 5]
'''



def g(x):
    return np.e**(-(x**2))

x0 = -5
x1 = 5
M = 1
N = 1000000

x =  np.linspace(x0+0.01,x1-0.01,100000)


plt.figure()
plt.title('Function to integrate: $g(x) = e^{-x^2}$')
plt.plot(x,g(x))
plt.xlabel('x')
plt.ylabel('g(x)')
plt.show()

Int = IntMonteCarlo(g,x0,x1,N,M)
print('Gaussian Integral in [-5,5] with N = 1,000,000 is: {}, with error {}'.format(round(Int[0],10),round(Int[1],7)))


Ii = []
Ni = []
Si = []


for k in np.logspace(3,6.5,100):
    Int = IntMonteCarlo(g,x0,x1,round(k),M)
    Ii.append(Int[0])
    Si.append(Int[1])
    Ni.append(round(k))

Ii = np.array(Ii)
Ni = np.array(Ni)
Si = np.array(Si)

plt.figure()
plt.title('Integral value of $g(x) = e^{-x^2}$ vs Number of steps')
plt.errorbar(Ni,Ii,yerr=Si, fmt = 'ro', ms=4, ecolor='r', elinewidth = 1,capsize = 4, capthick = 1)
plt.plot(Ni,Ii)
plt.ylabel('$\int gdx$')
plt.xlabel('Number of steps')
plt.semilogx()
plt.show()

PrintMonteCarloValores(g,x0,x1,N,M,'Monte Carlo Integration: Accepted Points for $g(x)$')




'''
Hyper-volume of a D-dimensional Sphere
Using Monte Carlo to estimate volume in high dimensions.
$V_D = \int ... \int dx_1 ... dx_D$ where $\sum x_i^2 \le R^2$
'''

def VolDEsfera(D,N):
    # Generate N points in a D-dimensional hypercube [-1, 1]^D
    x = rand.uniform(np.array(D*[-1]),np.array(D*[1]),(N,D))
    
    # Count points inside the unit hypersphere (norm < 1)
    NF = np.sum(np.linalg.norm(x,axis = 1) < 1)
    
    A = 2**D # Volume of the bounding hypercube
    I = A*NF/N
    s = np.sqrt(I*(A-I)/N)
    return I,s

I10,S10 = VolDEsfera(10,10000000)

I2,S2 = VolDEsfera(2,1000000)

print('Area of a unit circle (D=2): {}, with error {}'.format(round(I2,10),round(S2,7)))
print('Hyper-volume of a unit sphere in D=10: {}, with error {}'.format(round(I10,10),round(S10,7)))