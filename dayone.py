# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:00:00 2023

@author: ryanw
"""

import pytreegrav as ptg
import numpy as np
import matplotlib.pyplot as plt
import Galaxy

# %pylab
from pytreegrav import Accel, Potential

def GenerateICs(N,seed=42):
    np.random.seed(seed) # seed the RNG for reproducibility
    pos = np.random.normal(size=(N,3)) # positions of particles
    pos -= np.average(pos, axis=0) # put center of mass at the origin
    vel = np.zeros_like(pos) # initialize at rest
    vel -= np.average(vel, axis=0) # make average velocity 0
    softening = np.repeat(0.1, N) # initialize softening to 0.1
    masses = np.repeat(1./N, N) # make the system have unit mass
    return pos, masses, vel, softening

def collapseICs(N, vel_prop, seed=4080):
    ''' Initial conditions for a uniform collapse of N particles with initial velocity of vel_prop of the equilibrium velocity.
    '''
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
    pos -= np.average(pos, axis=0) # put center of mass at the origin
    des_vel = vel_prop * 0.5**(1/3)
    xprop = np.random.uniform(-1, 1, len(x))
    yprop = np.random.uniform(-1, 1, len(x))
    zprop = np.random.uniform(-1, 1, len(x))
    mult = np.sqrt(des_vel**2 / (xprop**2 + yprop**2 + zprop**2))
    xprop *= mult; yprop *= mult; zprop *= mult
    vel = np.zeros_like(pos) # initialize at rest
    vel[:, 0] = xprop; vel[:, 1] = yprop; vel[:, 2] = zprop
    
    softening = np.repeat(0.2, N) # initialize softening to 0.1\
    masses = np.repeat(1./N, N) # make the system have unit mass
    return pos, masses, vel, softening

def diskgalICs():
    ''' Initial conditions from Holger's disk galaxy. 
    '''
    data = np.genfromtxt('treecode/discgal.dat')
    pos = data[:, 1:4]
    vel = data[:, 4:7]
    N = len(pos[:, 0])
    softening = np.repeat(0.1, N) # initialize softening to 0.1
    masses = np.repeat(1./N, N) # make the system have unit mass
    return pos, masses, vel, softening

def DLGalICs(species):
    ''' Initial conditions from a Hubble classification 'species' galaxy from my distance ladder Universe Sim code. 
    '''
    BHpos = np.array([0, 0, 0])
    galaxy = Galaxy.Galaxy(species, BHpos, cartesian=True, blackhole=True, darkmatter=False, rotate=False)
    x, y, z, colours, scales = galaxy.starpositions
    
    colours = np.append(colours, np.array([galaxy.blackhole.get_BH_colour()]), axis=0)
    scales = np.append(scales, galaxy.blackhole.get_BH_scale())
    N = len(x) + 1
    pos = np.zeros((N, 3))
    pos[:, 0] = np.append(x, BHpos[0]); pos[:, 1] = np.append(y, BHpos[1]); pos[:, 2] = np.append(z, BHpos[2])
    pos /= galaxy.radius
    
    velmags, directions = galaxy.starvels, galaxy.directions
    velmags = velmags[:len(velmags)//2]
    vel = np.zeros((N, 3))
    vel[:, 0] = np.append(directions[0, :], 0); vel[:, 1] = np.append(directions[1, :], 0); vel[:, 2] = np.append(directions[2, :], 0)
    vel = (vel.T * np.append(velmags, 0)).T
    
    vel /= np.sqrt(6.67 * 10**-11 * galaxy.galaxymass * 1.98 * 10**30 / (galaxy.radius * 4/3 * 3.086 * 10**16))
    vel *= 9e2
    
    masses = np.append(galaxy.starmasses, galaxy.galaxyBHmass) / galaxy.galaxymass
    softening = np.repeat(0.1, N) # initialize softening to 0.1
    return pos, masses, vel, softening, colours, scales

def TotalEnergy(pos, masses, vel, softening):
    kinetic = 0.5 * np.sum(masses[:,None] * vel**2)
    potential = 0.5 * np.sum(masses * Potential(pos,masses,softening,parallel=True))
    return kinetic + potential

def leapfrog_kdk_timestep(dt, pos, masses, softening, vel, accel):
    # first a half-step kick
    vel[:] = vel + 0.5 * dt * accel # note that you must slice arrays to modify them in place in the function!
    # then full-step drift
    pos[:] = pos + dt * vel - np.median(pos)
    # then recompute accelerations
    accel[:] = Accel(pos,masses,softening,parallel=True)
    # then another half-step kick
    vel[:] = vel + 0.5 * dt * accel

N = int(4096)
# pos, masses, vel, softening = GenerateICs(N) # initialize initial condition with 10k particles
pos, masses, vel, softening = collapseICs(N, 0.05, seed=3080)
# pos, masses, vel, softening = diskgalICs()
# pos, masses, vel, softening, colours, scales = DLGalICs('Sa')
N = len(pos[:, 0])

accel = Accel(pos,masses,softening,parallel=True) # initialize acceleration

t = 0 # initial time
dt = 0.02 # adjust this to control integration error
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

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, marker='.')
length = 15
fps = nt / length
xmin, xmax = min(positions[:, 0, 0]), max(positions[:, 0, 0])
ymin, ymax = min(positions[:, 1, 0]), max(positions[:, 1, 0])
zmin, zmax = min(positions[:, 2, 0]), max(positions[:, 2, 0])

limmin, limmax = min([xmin, ymin, zmin]), max([xmax, ymax, zmax])
ax.set_facecolor('k')
def animate(i):
    # print(i)
    ax.clear()
    # ax.scatter(positions[:, 0, i], positions[:, 1, i], positions[:, 2, i], s=scales, marker='.', c=colours)
    ax.scatter(positions[:, 0, i], positions[:, 1, i], positions[:, 2, i], s=1, marker='.', c='w')
    ax.set_xlim(limmin, limmax); ax.set_ylim(limmin, limmax); ax.set_zlim(limmin, limmax)
    
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=nt, interval=int(1000/fps), cache_frame_data=False)


# plt.show()

ani.save(f'Uniform Collapse.gif', writer='pillow')
# ani.save(f'Diskgal.gif', writer='pillow')
# ani.save(f'GALAXY.gif', writer='pillow')

# plt.close('all')