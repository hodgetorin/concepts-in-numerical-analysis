# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:28:38 2021

@author: Torin
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt

C_0 = 1
mu = 0.1

dt = 1 # time step
dt_0 = 0 # initial time step
dt_count = 10 # number of time steps

C = np.ones(dt_count + 1) # empty array for our exponential approximation
C_exp = np.ones(dt_count +1) # empty array for our Euler approximation
C_taylor = np.ones(dt_count+1) # empty array for our taylor series approximation


time = np.zeros(dt_count+1) # empty array for time

# exponential:
for i in range (1,dt_count+1):
    time[i] = dt_0 + i*dt
    dc = mu*dt*C[i-1]
    C[i] = C[i-1] + dc
    
# euler:
for i in range (1, dt_count +1):
    C_exp[i] = C_0*exp(mu*time[i])

# taylor:
for i in range (1, dt_count+1):
    C_taylor[i] = C_0 + mu*C_0*time[i] + (mu**2)*C_0*((time[i]**2)/2)
    

plt.plot(time, C, 'ro', time, C_exp, C_taylor, 'bo')
plt.xlabel("Day"); plt.ylabel("AFDW g/L")
plt.legend(['Euler', 'Exponential', 'Taylor Approximation'])