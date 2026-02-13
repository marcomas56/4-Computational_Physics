import numpy as np
import matplotlib.pyplot as plt
import time

'''
Solving Ordinary Differential Equations (ODEs)
Marco Mas Sempere
System: $\ddot{x} + a\dot{x} + \omega^2 x = 0$
Matrix form: $\dot{\mathbf{u}} = A\mathbf{u}$
'''

#Defining the constants of our equation
w = 1
a = 0.2

#Defining the matrices needed
A = np.array([[0,1],[-w**2,-a]])
I = np.array([[1,0],[0,1]])

#Defining initial values
x0 = 1
v0 = 0.4

#Discretization of the time interval
n = 5000
tf = 50
ti = 0
t = np.linspace(ti,tf,n)
dt = t[1]-t[0]


#Function to calculate the analytical solution
def sol(t) -> np.ndarray:
    if a**2 < 4*w**2: # Underdamped
        omega_prima = np.sqrt(w**2 - (a**2) / 4)
        C1 = x0
        C2 = (v0 + a * x0 / 2) / omega_prima
        x_analytical = np.exp(-a * t / 2) * (C1 * np.cos(omega_prima * t) + C2 * np.sin(omega_prima * t))
            
    elif a**2 == 4*w**2: # Critically damped
        C1 = x0
        C2 = v0 + a/2 * x0
        x_analytical = (C1 + C2 * t) * np.exp(-a * t / 2)
            
    elif a**2 > 4*w**2: # Overdamped
        gamma = np.sqrt((a**2) / 4 - w**2)
        C1 = (x0 / 2) + (v0 + (a / 2) * x0) / (2 * gamma)
        C2 = (x0 / 2) - (v0 + (a / 2) * x0) / (2 * gamma)
        x_analytical = np.exp(-a * t / 2) * (C1 * np.exp(gamma * t) + C2 * np.exp(-gamma * t))
    return x_analytical

sol = np.vectorize(sol)


'''
Euler Forward Method
$U_{n+1} = (I + \Delta t A) U_n$
'''


def Euler_Forward(x0,v0,ti,tf,n) -> np.ndarray:
    '''
    Function that solves using the Euler Forward method
    '''

    t = np.linspace(ti,tf,n)
    dt = t[1]-t[0]

    U = np.zeros((2,n))
    U[:,0] = [x0,v0]

    for i in range(0,n-1):
        U[:,i+1] = np.dot((I+dt*A),U[:,i])
    return U

xF = Euler_Forward(x0,v0,ti,tf,n)[0,:]


#Plotting comparison with the analytical solution
plt.figure('Euler Forward Method')
plt.title('Euler Forward Method')
plt.plot(t,sol(t),label = 'Analytical Solution')
plt.plot(t,xF,label = 'Numerical Solution')
plt.legend()
plt.ylabel('x(t)')
plt.xlabel('t')



'''
Euler Backward Method
$U_{n+1} = (I - \Delta t A)^{-1} U_n$
'''

def Euler_Backward(x0,v0,ti,tf,n) -> np.ndarray:
    '''
    Function that solves using the Euler Backward method
    '''

    t = np.linspace(ti,tf,n)
    dt = t[1]-t[0]

    U = np.zeros((2,n))
    U[:,0] = [x0,v0]

    for i in range(0,n-1):
        U[:,i+1] = np.dot(np.linalg.inv((I-dt*A)),U[:,i])
    return U

xB = Euler_Backward(x0,v0,ti,tf,n)[0,:]

#Plotting comparison with the analytical solution
plt.figure('Euler Backward Method')
plt.title('Euler Backward Method')
plt.plot(t,sol(t),label = 'Analytical Solution')
plt.plot(t,xB,label = 'Numerical Solution')
plt.legend()
plt.ylabel('x(t)')
plt.xlabel('t')




'''
Semi-implicit Method (Crank-Nicolson)
$U_{n+1} = (I - \frac{\Delta t}{2} A)^{-1} (I + \frac{\Delta t}{2} A) U_n$
'''

def Semi_implicito(x0,v0,ti,tf,n) -> np.ndarray:
    '''
    Function that solves using the Semi-Implicit method
    '''


    t = np.linspace(ti,tf,n)
    dt = t[1]-t[0]

    U = np.zeros((2,n))
    U[:,0] = [x0,v0]

    for i in range(0,n-1):
        U[:,i+1] = np.dot(np.dot(np.linalg.inv(I-(dt/2)*A),I+(dt/2)*A),U[:,i])
    return U

xS = Semi_implicito(x0,v0,ti,tf,n)[0,:]

#Plotting comparison with the analytical solution
plt.figure('Semi-implicit Method')
plt.title('Semi-implicit Method')
plt.plot(t,sol(t),label = 'Analytical Solution')
plt.plot(t,xS,label = 'Numerical Solution')
plt.legend()
plt.ylabel('x(t)')
plt.xlabel('t')


'''
Runge-Kutta (RK4) Method
'''

def Runge_Kutta4(x0,v0,ti,tf,n) -> np.ndarray:
    '''
    Function that solves using Runge-Kutta 4
    '''
    t = np.linspace(ti,tf,n)
    dt = t[1]-t[0]

    U = np.zeros((2,n))
    U[:,0] = [x0,v0]


    for i in range(0,n-1):
        y1 = np.dot(A,U[:,i])
        y2 = np.dot(A,U[:,i]+(dt/2)*y1)
        y3 = np.dot(A,U[:,i]+(dt/2)*y2)
        y4 = np.dot(A,U[:,i]+dt*y3)
        
        U[:,i+1] = U[:,i] + (y1 + 2*y2 + 2*y3 + y4)*dt/6
    return U

xK = Runge_Kutta4(x0,v0,ti,tf,n)[0,:]

#Plotting comparison with the analytical solution
plt.figure('Runge-Kutta 4 Method')
plt.title('Runge-Kutta 4 Method')
plt.plot(t,sol(t),label = 'Analytical Solution')
plt.plot(t,xK,label = 'Numerical Solution')
plt.legend()
plt.ylabel('x(t)')
plt.xlabel('t')


'''
Error Comparison
'''

#Defining a norm to calculate errors
def Norma_l2(a,b,dt) -> float:
    '''
    Calculates the discrete L2 norm
    '''

    return np.sqrt(np.sum((a-b)**2*dt))

s = sol(t)


#Comparison of errors as a function of t for the 4 methods

plt.figure('Comparison of errors vs t')
plt.title('Comparison of errors vs t')
plt.plot(t,np.abs(xF-s),label = 'Euler Forward Error')
plt.plot(t,np.abs(xB-s),label = 'Euler Backward Error')
plt.plot(t,np.abs(xS-s),label = 'Semi-Implicit Error')
plt.plot(t,np.abs(xK-s),label = 'Runge-Kutta 4 Error')
plt.legend()
plt.ylabel('Error')
plt.xlabel('t')



#Comparison of errors using L2 norm as a function of dt for the 4 methods

N = [50,100,250,500,1000,2500,5000,10000,25000,50000,100000]
dT = []
EF = []
EB = []
ES = []
EK = []


for n in N:

    t = np.linspace(ti,tf,n)
    dt = t[1]-t[0]
    dT.append(dt)

    s = sol(t)

    xF = Euler_Forward(x0,v0,ti,tf,n)[0,:]
    EF.append(Norma_l2(xF,s,dt))

    xB = Euler_Backward(x0,v0,ti,tf,n)[0,:]
    EB.append(Norma_l2(xB,s,dt))


    xS = Semi_implicito(x0,v0,ti,tf,n)[0,:]
    ES.append(Norma_l2(xS,s,dt))


    xK = Runge_Kutta4(x0,v0,ti,tf,n)[0,:]
    EK.append(Norma_l2(xK,s,dt))


dT = np.array(dT)
EF = np.array(EF)
EB = np.array(EB)
ES = np.array(ES)
EK = np.array(EK)


plt.figure('Comparison of errors vs dt')
plt.title('Comparison of errors vs dt')
plt.plot(dT,EF,label = 'Euler Forward Error')
plt.plot(dT,EB,label = 'Euler Backward Error')
plt.plot(dT,ES,label = 'Semi-Implicit Error')
plt.plot(dT,EK,label = 'Runge-Kutta 4 Error')
plt.semilogy()
plt.semilogx()
plt.legend()
plt.ylabel('Error')
plt.xlabel('dt')
plt.show()