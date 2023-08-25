## if you work remotely, uncomment the following two lines
##import matplotlib as mpl
## mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy import *

f2 = open('data', 'r') 
stime=float(f2.readline())
lines = f2.readlines()
x = []
y = [] 
for line in lines:
    p = line.split()
    x.append(float(p[1]))
    y.append(float(p[2]))
f2.close() 

plt.figure(figsize = (7,7))
plt.tick_params(left = False, bottom = False)

plt.xlim([-20, 20])
plt.ylim([-20, 20])

plt.plot(x, y, color='r', linestyle='None')
plt.grid(False)
plt.scatter(x, y)
timestr = "Time = {}".format(stime)
plt. title(timestr) 
##plt.show()
plt.savefig('snap.png')
