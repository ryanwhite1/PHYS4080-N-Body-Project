# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:00:00 2023

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import CommonTools as nbody
import matplotlib

plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.fontset']='cm'

N = 4096
Tmax = 5
dt = 0.02
nt = int((Tmax - 0) / dt) + 1
times = np.linspace(0, Tmax, nt)

### lets first plot the equilibrium condition for the halo
pos, masses, vel, softening = nbody.collapseICs(N, 1)
positions = nbody.perform_sim(Tmax, dt, pos, masses, vel, softening)

# now to choose the colours for the points
radii = np.sqrt(positions[:, 0, 0]**2 + positions[:, 1, 0]**2 + positions[:, 2, 0]**2)
norm = matplotlib.colors.Normalize(vmin=min(radii), vmax=max(radii))
norm_radii = norm(radii)
cmap = matplotlib.colormaps['autumn']
colours = cmap(norm_radii)

nbody.animate_sim(positions, 'Equilibrium', 12, colours=colours, every=1, times=[True, times])


### now lets plot the cold collapse
pos, masses, vel, softening = nbody.collapseICs(N, 0.05)
positions = nbody.perform_sim(Tmax, dt, pos, masses, vel, softening)

# now to choose the colours for the points
radii = np.sqrt(positions[:, 0, 0]**2 + positions[:, 1, 0]**2 + positions[:, 2, 0]**2)
norm = matplotlib.colors.Normalize(vmin=min(radii), vmax=max(radii))
norm_radii = norm(radii)
cmap = matplotlib.colormaps['autumn']
colours = cmap(norm_radii)

nbody.animate_sim(positions, 'ColdCollapse', 12, colours=colours, every=1, times=[True, times])


### now we can start plotting graphs of the behaviour due to the cold collapse
# first lets get the 10, 20, 50, and 90% spheres of the particles
radii = np.zeros((nt, 4))
for i in range(nt):
    radii[i, 0] = nbody.prop_sphere(0.1, positions[:, :, i])
    radii[i, 1] = nbody.prop_sphere(0.2, positions[:, :, i])
    radii[i, 2] = nbody.prop_sphere(0.5, positions[:, :, i])
    radii[i, 3] = nbody.prop_sphere(0.9, positions[:, :, i])
  

# and now lets plot these radii of the 10, etc % spheres over time
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvline(np.pi / (2 * np.sqrt(2)), c='k', ls='--', label='$t_{ff}$')
ax.plot(times, radii[:, 0], label='10\% Sphere')
ax.plot(times, radii[:, 1], label='20\% Sphere')
ax.plot(times, radii[:, 2], label='50\% Sphere')
ax.plot(times, radii[:, 3], label='90\% Sphere')
ax.legend()
ax.set_yscale('log')
ax.set_xlim(0, Tmax)
ax.set_xlabel("Time ($N$-body units)")
ax.set_ylabel("Radius ($N$-body units)")

fig.savefig('ColdCollapse.png', dpi=400, bbox_inches='tight')
fig.savefig('ColdCollapse.pdf', dpi=400, bbox_inches='tight')

### Now convert to real units for a Milky Way like halo
MWmass = 10e12
MWrad = 100e3
real_time = nbody.time_convert(times, MWmass, MWrad)
real_rad = radii * MWrad / 1e3

fig, ax = plt.subplots(figsize=(8, 5))
ax.axvline(nbody.free_fall_time(MWrad, MWmass), c='k', ls='--', label='$t_{ff}$')
ax.plot(real_time, real_rad[:, 0], label='10\% Sphere')
ax.plot(real_time, real_rad[:, 1], label='20\% Sphere')
ax.plot(real_time, real_rad[:, 2], label='50\% Sphere')
ax.plot(real_time, real_rad[:, 3], label='90\% Sphere')
ax.legend()
ax.set_yscale('log')
ax.set_xlim(0, max(real_time))
ax.set_xlabel("Time (Myr)")
ax.set_ylabel("Radius (kpc)")

fig.savefig('ColdCollapse-MWHalo.png', dpi=400, bbox_inches='tight')
fig.savefig('ColdCollapse-MWHalo.pdf', dpi=400, bbox_inches='tight')

plt.close('all')

