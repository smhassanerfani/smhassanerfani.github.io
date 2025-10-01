# Numerical Solution with Euler Method
import numpy as np

# ODE: dy/dt = 2t - y
def f(t, y):
    return 2*t - y

# Euler method implementation
def euler_method(f, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return y

# Time grid and solution
t = np.linspace(0, 5, 100)  # from t=0 to t=5 with 100 steps
y_euler = euler_method(f, y0=-1, t=t)