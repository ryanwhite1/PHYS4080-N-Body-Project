# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:34:23 2023

@author: ryanw
"""

import pytreegrav as ptg
import numpy as np
import matplotlib.pyplot as plt
import CommonTools as nbody

plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.fontset']='cm'

Tmax = 100
dt = 0.1
nt = int((Tmax - 0) / dt) + 1
pos1, masses1, vel1, softening1 = nbody.diskgalICs()
N1 = len(pos1[:, 0])
N = 2 * N1

colours = np.append(['lightcoral' for _ in range(N1)],['paleturquoise' for _ in range(N1)])

rel_vels = [4, 2.25, 0.5]
for rel_vel in rel_vels:
    pos = np.append(pos1, pos1, axis=0)
    masses = np.append(masses1, masses1) / 2    # divide by two so the total mass of the system is still =1
    vel = np.append(vel1, vel1, axis=0)
    softening = np.append(softening1, softening1)
    
    # separate the galaxies by Dx = 30, and Dy = 5, while keeping center of mass at 0
    pos[:N1, 0] += -15; pos[N1:, 0] += 15
    pos[:N1, 1] += -2.5; pos[N1:, 1] += 2.5
    
    escape_vel = np.sqrt(2 / np.sqrt(15**2 + 2.5**2))
    
    vel[:N1, 0] += (rel_vel / 2) * escape_vel; vel[N1:, 0] += - (rel_vel / 2) * escape_vel
    
    positions = nbody.perform_sim(Tmax, dt, pos, masses, vel, softening)
    
    nbody.animate_sim(positions, f'GalaxyCollision-{rel_vel}v_e', 8, colours=colours)

