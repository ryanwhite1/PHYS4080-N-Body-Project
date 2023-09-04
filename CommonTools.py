# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:00:00 2023

@author: ryanw
"""

import pytreegrav as ptg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

def collapseICs(N, vel_prop, seed=4080):
    ''' Initial conditions for a uniform collapse of N particles with initial velocity of vel_prop of the equilibrium velocity.
    '''
    np.random.seed(seed) # seed the RNG for reproducibility
    
    # uniformly (and randomly) distribute points in the unit sphere
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
    pos -= np.average(pos, axis=0) # apply small correction to put center of mass at the origin
    
    # now to give particles random velocity with a magnitude of 'vel_prop' of the equilibrium velocity
    des_vel = vel_prop * 0.5**(1/3)
    xprop = np.random.uniform(-1, 1, len(x))
    yprop = np.random.uniform(-1, 1, len(x))
    zprop = np.random.uniform(-1, 1, len(x))
    mult = np.sqrt(des_vel**2 / (xprop**2 + yprop**2 + zprop**2))
    xprop *= mult; yprop *= mult; zprop *= mult
    vel = np.zeros_like(pos) # initialize at rest
    vel[:, 0] = xprop; vel[:, 1] = yprop; vel[:, 2] = zprop
    vel -= np.average(vel, axis=0) # make average velocity 0
    
    softening = np.repeat(0.2, N) if N > 4e3 else np.repeat(0.1, N)
    masses = np.repeat(1./N, N) # make the system have unit mass
    return pos, masses, vel, softening

def diskgalICs():
    ''' Initial conditions from Holger's disk galaxy. 
    '''
    data = np.genfromtxt('treecode/discgal.dat')
    pos = data[:, 1:4]
    vel = data[:, 4:7]
    N = len(pos[:, 0])
    softening = np.repeat(0.15, N) # initialize softening to 0.1
    masses = np.repeat(1./N, N) # make the system have unit mass
    return pos, masses, vel, softening

def TotalEnergy(pos, masses, vel, softening):
    kinetic = 0.5 * np.sum(masses[:, None] * vel**2)
    potential = 0.5 * np.sum(masses * ptg.Potential(pos, masses, softening, parallel=True))
    return kinetic + potential

def leapfrog_kdk_timestep(dt, pos, masses, softening, vel, accel):
    '''
    '''
    # first a half-step kick
    vel[:] = vel + 0.5 * dt * accel # note that you must slice arrays to modify them in place in the function!
    # then full-step drift
    pos[:] = pos + dt * vel
    # then recompute accelerations
    accel[:] = ptg.Accel(pos, masses, softening, parallel=True)
    # then another half-step kick
    vel[:] = vel + 0.5 * dt * accel
    
def animate_sim(positions, filename, length, colours=[], every=1, times=[False]):
    ''' Animates the positions of N points in 3D space for nt timesteps against a black background, and saves it too!
    Parameters
    ----------
    positions : (N x 3 x nt) ndarray
        Particle position in xyz space for each of the N particles at each of the nt time steps. 
    filename : str
        The desired filename, to be saved as 'filename.gif'
    length : float
        Desired length (in seconds) of the gif
    colours : Nx1 list/array (optional)
        The (order dependent) colours to plot each of the N data points
    every : int
        Will plot every n frames in the animation. every=1 corresponds to plotting each frame, every=2 each second frame, etc.
    times : list
        Either 1 element list (containing just False) if we don't want a little timer in the top right, or, 
        a 2 element list (the first being True) with the second element containing an nt x 1 array of times. 
    '''
    fig = plt.figure(figsize=(12, 12), frameon=False)   # we want no frame so that it's a clean black animation
    ax = fig.add_subplot(projection='3d')   # 3d axes
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) # removes (most of the) blank border around the plot
    # now do an initial scatter so we can get the axis limits to use through the animation
    ax.scatter(positions[:, 0, 0], positions[:, 1, 0], positions[:, 2, 0], s=1, marker='.')
    xmin, xmax = min(positions[:, 0, 0]), max(positions[:, 0, 0])
    ymin, ymax = min(positions[:, 1, 0]), max(positions[:, 1, 0])
    zmin, zmax = min(positions[:, 2, 0]), max(positions[:, 2, 0])
    limmin, limmax = min([xmin, ymin, zmin]), max([xmax, ymax, zmax])   # get the minimum and maximums for the axis limits
    
    # now calculate some parameters for the animation frames and timing
    nt = len(positions[0, 0, :]) # number of timesteps
    frames = np.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) / length  # fps for the final animation
    
    ax.set_facecolor('k')   # black background, since space is blach duh
    
    def animate(i):
        if (i // every)%20 == 0:
            print(f"{i // every} / {len(frames)}")
        ax.clear()
        if len(colours) != 0:
            ax.scatter(positions[:, 0, i], positions[:, 1, i], positions[:, 2, i], s=1, marker='.', 
                       c=colours)
        else:
            ax.scatter(positions[:, 0, i], positions[:, 1, i], positions[:, 2, i], s=1, marker='.', c='w')
        ax.set_xlim(limmin, limmax); ax.set_ylim(limmin, limmax); ax.set_zlim(limmin, limmax)
        if times[0]:    # plot the current time in the corner if we want to!
            ax.text(0.7 * limmax, 0.7 * limmax, 1.85 * limmax, "$T = " + str(round(times[1][i], 1)) + "$", fontsize=24, color='w')
        
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        return fig,

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=1, cache_frame_data=False, blit=True)
    ani.save(f"{filename}.gif", writer='pillow', fps=fps)
    plt.close('all')

def perform_sim(Tmax, dt, pos, masses, vel, softening):
    ''' Performs an nbody sim from t=0 to t=Tmax [in steps of dt] given particles with parameters
    pos : (N x 3) ndarray
        Positions of N particles in xyz coordinates
    masses : (N x 1) array
        Mass of each particle
    vel : (N x 3) ndarray
        Velocities of N particles in xyz components
    softening : (N x 1) array
        Softening coefficient of each particle
    Returns
    -------
    positions : (N x 3 x nt) ndarray
        Particle position in xyz space for each of the N particles at each of the nt time steps. 
    '''
    t1 = time.time()
    N = len(pos[:, 0])
    accel = ptg.Accel(pos, masses, softening, parallel=True) # initialize acceleration

    t = 0 # initial time
    nt = int((Tmax - t) / dt) + 1

    energies = np.zeros(nt)

    positions = np.zeros((N, 3, nt))
    positions[:, :, 0] = pos
    
    i = 0
    while t <= Tmax: # actual simulation loop - this may take a couple minutes to run
        energies[i] = TotalEnergy(pos, masses, vel, softening)
        
        leapfrog_kdk_timestep(dt, pos, masses, softening, vel, accel)
        positions[:, :, i] = pos
        t += dt
        i += 1
    t2 = time.time()
    print(f"Simulation complete in {round(t2 - t1, 3)}s! Relative energy error: {(np.abs((energies[0]-energies[-1])/energies[0]))}")
    return positions
    
def prop_sphere(prop, positions):
    ''' Finds the radius of the sphere containing `prop` * 100% of the same mass particles.
    '''
    return np.percentile(np.sum((positions - np.median(positions, axis=0))**2, axis=1)**0.5, prop * 100)

def time_convert(time, M, R):
    ''' Converts from n-body time to real time (in units of Myr).
    Parameters
    ----------
    time : float/array
        N-body times
    M : float
        Mass of the units in solar masses
    R : float
        Radius of the system in pc
    '''
    mass = M * 1.988 * 10**30
    radius = R * 3.086 * 10**16
    G = 6.6743 * 10**-11
    Myr_sec = 31536000000000.0
    return time * np.sqrt(radius**3 / (mass * G)) / Myr_sec

def free_fall_time(radius, mass):
    ''' Returns free fall time in Myr.
    Parameters
    ----------
    M : float
        Mass of the units in solar masses
    R : float
        Radius of the system in pc
    '''
    M = mass * 1.988 * 10**30
    R = radius * 3.086 * 10**16
    G = 6.6743 * 10**-11
    Myr_sec = 31536000000000.0
    return ((np.pi / 2) * R**(3/2) / np.sqrt(2 * G * M)) / Myr_sec

def com_sep(pos1, pos2):
    ''' Center of mass separation.
    Parameters
    ----------
    pos1, pos2 : (Nx3) np.array
    '''
    com1 = np.mean(pos1, axis=0)
    com2 = np.mean(pos2, axis=0)
    return np.linalg.norm(com1 - com2)






