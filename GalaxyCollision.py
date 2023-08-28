# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:34:23 2023

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import CommonTools as nbody

plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.fontset']='cm'

Tmax = 100
dt = 0.05
nt = int((Tmax - 0) / dt) + 1
times = np.linspace(0, Tmax, nt)

pos1, masses1, vel1, softening1 = nbody.diskgalICs()
N1 = len(pos1[:, 0])
N = 2 * N1

colours = np.append(['lightcoral' for _ in range(N1)],['paleturquoise' for _ in range(N1)])

all_pos = np.zeros((2 * N1, 3, nt, 3))

rel_vels = [4, 2.25, 0.5]
for i, rel_vel in enumerate(rel_vels):
    pos = np.append(pos1, pos1, axis=0)
    masses = np.append(masses1, masses1)
    vel = np.append(vel1, vel1, axis=0)
    softening = np.append(softening1, softening1)
    
    # separate the galaxies by Dx = 30, and Dy = 5, while keeping center of mass at 0
    pos[:N1, 0] += -15; pos[N1:, 0] += 15
    pos[:N1, 1] += -2.5; pos[N1:, 1] += 2.5
    
    escape_vel = np.sqrt(2 * 2 / np.sqrt(15**2 + 2.5**2))
    
    vel[:N1, 0] += (rel_vel / 2) * escape_vel; vel[N1:, 0] += - (rel_vel / 2) * escape_vel
    
    positions = nbody.perform_sim(Tmax, dt, pos, masses, vel, softening)
    
    all_pos[:, :, :, i] = positions
    
    nbody.animate_sim(positions, f'GalaxyCollision-{rel_vel}v_e', 8, colours=colours)
    
    # we want to plot the radial extent of stellar populations of one galaxy in the final merger case
    if i == len(rel_vels) - 1:
        radii = np.zeros((nt, 3))
        for i in range(nt):
            radii[i, 0] = nbody.prop_sphere(0.1, positions[:N1, :, i])
            radii[i, 1] = nbody.prop_sphere(0.2, positions[:N1, :, i])
            radii[i, 2] = nbody.prop_sphere(0.5, positions[:N1, :, i])
          

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, radii[:, 0], label='10\% Sphere')
        ax.plot(times, radii[:, 1], label='20\% Sphere')
        ax.plot(times, radii[:, 2], label='50\% Sphere')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlim(0, Tmax)
        ax.set_xlabel("Time ($N$-body units)")
        ax.set_ylabel("Radius ($N$-body units)")

        fig.savefig(f'GalaxyCollision-{rel_vel}v_e-StarRadii.png', dpi=400, bbox_inches='tight')
        fig.savefig(f'GalaxyCollision-{rel_vel}v_e-StarRadii.pdf', dpi=400, bbox_inches='tight')
    

# now to plot the center of mass separations of the galaxies over time
com_separations = np.zeros((nt, len(rel_vels)))

for i in range(nt):
    for j in range(len(rel_vels)):
        com_separations[i, j] = nbody.com_sep(all_pos[:N1, :, i, j], all_pos[N1:, :, i, j])
    

fig, ax = plt.subplots(figsize=(8, 5))

for i in range(3):
    ax.plot(times, com_separations[:, i], label=f'$v_i = {rel_vels[i]}v_e$')
ax.legend()
ax.set_xlabel("Time ($N$-body units)")
ax.set_ylabel("CoM Separation ($N$-body units)")
ax.set_yscale('log')
fig.savefig("GalaxyCollision-CoMSep.png", dpi=400, bbox_inches='tight')
fig.savefig("GalaxyCollision-CoMSep.pdf", dpi=400, bbox_inches='tight')
    


