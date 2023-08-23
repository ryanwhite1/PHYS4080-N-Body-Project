# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:00:00 2023

@author: ryanw
"""

import pytreegrav as ptg
import numpy as np
import matplotlib.pyplot as plt

# %pylab
from pytreegrav import Accel, Potential

def GenerateICs(N,seed=42):
    np.random.seed(seed) # seed the RNG for reproducibility
    pos = np.random.normal(size=(N,3)) # positions of particles
    pos -= np.average(pos,axis=0) # put center of mass at the origin
    vel = np.zeros_like(pos) # initialize at rest
    vel -= np.average(vel,axis=0) # make average velocity 0
    softening = np.repeat(0.1,N) # initialize softening to 0.1
    masses = np.repeat(1./N,N) # make the system have unit mass
    return pos, masses, vel, softening

def TotalEnergy(pos, masses, vel, softening):
    kinetic = 0.5 * np.sum(masses[:,None] * vel**2)
    potential = 0.5 * np.sum(masses * Potential(pos,masses,softening,parallel=True))
    return kinetic + potential

def leapfrog_kdk_timestep(dt, pos, masses, softening, vel, accel):
    # first a half-step kick
    vel[:] = vel + 0.5 * dt * accel # note that you must slice arrays to modify them in place in the function!
    # then full-step drift
    pos[:] = pos + dt * vel
    # then recompute accelerations
    accel[:] = Accel(pos,masses,softening,parallel=True)
    # then another half-step kick
    vel[:] = vel + 0.5 * dt * accel

N = 1000
pos, masses, vel, softening = GenerateICs(N) # initialize initial condition with 10k particles

accel = Accel(pos,masses,softening,parallel=True) # initialize acceleration

t = 0 # initial time
dt = 0.03 # adjust this to control integration error
Tmax = 50 # final/max time
nt = int((Tmax - t) / dt) + 1

energies = [] #energies
r50s = [] #half-mass radii
ts = [] # times

positions = np.zeros((N, 3, nt))
positions[:, :, 0] = pos


i = 0
while t <= Tmax: # actual simulation loop - this may take a couple minutes to run
    r50s.append(np.median(np.sum((pos - np.median(pos,axis=0))**2,axis=1)**0.5))
    energies.append(TotalEnergy(pos,masses,vel,softening))
    ts.append(t)
    

    leapfrog_kdk_timestep(dt, pos, masses, softening, vel, accel)
    positions[:, :, i] = pos
    t += dt
    i += 1
print("Simulation complete! Relative energy error: %g"%(np.abs((energies[0]-energies[-1])/energies[0])))



from matplotlib import animation

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, marker='.')
length = 20
fps = nt / length
xmin, xmax = min(positions[:, 0, 0]), max(positions[:, 0, 0])
ymin, ymax = min(positions[:, 1, 0]), max(positions[:, 1, 0])
zmin, zmax = min(positions[:, 2, 0]), max(positions[:, 2, 0])
def animate(i):
    print(i)
    ax.clear()
    ax.scatter(positions[:, 0, i], positions[:, 1, i], positions[:, 2, i], s=1, marker='.')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=nt, interval=int(1000/fps))


plt.show()

ani.save(f'test.gif', writer='pillow')

plt.close('all')