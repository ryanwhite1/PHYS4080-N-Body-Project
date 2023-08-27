# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:00:00 2023

@author: ryanw
"""

import pytreegrav as ptg
import numpy as np
import matplotlib.pyplot as plt
import CommonTools as nbody

N = 4096
Tmax = 10
dt = 0.02
pos, masses, vel, softening = nbody.collapseICs(N, 0.05)

positions = nbody.perform_sim(Tmax, dt, pos, masses, vel, softening)

nbody.animate_sim(positions, 'ColdCollapse', 15)
