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

def collapseICs(N, seed=4080):
    np.random.seed(seed) # seed the RNG for reproducibility
    
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.random.uniform(-1, 1, N)
    phi = np.arccos(phi)
    dists = np.random.uniform(0, 1, N)
    R = 1 * dists**(1/3)
    x = R * np.cos(theta) * np.sin(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(phi)
    pos = np.zeros((N, 3))
    pos[:, 0] = x; pos[:, 1] = y; pos[:, 2] = z
    pos -= np.average(pos,axis=0) # put center of mass at the origin
    des_vel = 0.05 * 1
    vel_x_comp = np.random.uniform(0, des_vel, N)
    vel_y_comp = np.random.uniform(0, des_vel - vel_x_comp, N)
    vel_z_comp = np.sqrt(des_vel**2 - (vel_x_comp**2 + vel_y_comp**2))
    vel = np.zeros_like(pos) # initialize at rest
    vel[:, 0] = vel_x_comp; vel[:, 1] = vel_y_comp; vel[:, 2] = vel_z_comp
    # vel -= np.average(vel,axis=0) # make average velocity 0
    softening = np.repeat(0.1,N) # initialize softening to 0.1
    masses = np.repeat(1./N,N) # make the system have unit mass
    return pos, masses, vel, softening

def diskgalICs():
    data = np.genfromtxt('treecode/discgal.dat')
    pos = data[:, 1:4]
    vel = data[:, 4:7]
    N = len(pos[:, 0])
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

N = int(1e3)
# pos, masses, vel, softening = GenerateICs(N) # initialize initial condition with 10k particles
# pos, masses, vel, softening = collapseICs(N)
pos, masses, vel, softening = diskgalICs()
N = len(pos[:, 0])

accel = Accel(pos,masses,softening,parallel=True) # initialize acceleration

t = 0 # initial time
dt = 0.03 # adjust this to control integration error
Tmax = 10 # final/max time
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

# ani.save(f'Uniform Collapse.gif', writer='pillow')
ani.save(f'Diskgal.gif', writer='pillow')

# plt.close('all')