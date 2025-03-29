#!/usr/bin/env python
# coding: utf-8

##### The script is written to compute swim (run) lengths and chord lengths. It outputs Figure 7 and Supplementary Fig 8. Last changed March 28, 2025.

import os
import glob,csv
import math
import scipy as sp
import numpy as np
import pandas as pd
#pip install -U phidl
import random
import phidl.geometry as pg
from phidl import quickplot as qp
from phidl import Device
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import KDTree
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

# ### Set the environment

### Define a function to compute chord lengths
def ChordLength(ctr, tree, radius, bounds, numSamples=100):
    rng = np.random.default_rng(12345)
    eps = 0.2  # 'step size'
    angles = rng.uniform(0.0, 2 * np.pi, numSamples)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    finalLen = np.zeros(numSamples)  # initialize a list to store the final chord lengths
    finalPos = []
    for angleCtr, (cos_angle, sin_angle) in enumerate(zip(cos_angles, sin_angles)):
        location = np.copy(ctr)
        collided = False
        while not collided:  # if the particle does not collide with a pillar
            location[0] += eps * cos_angle  # update the x position
            location[1] += eps * sin_angle  # update the y position

            # boundary consideration
            if location[0] > bounds[0] or location[1] > bounds[1] or location[0] < 0 or location[1] < 0:
                collided = True
                finalLen[angleCtr] = np.linalg.norm(ctr - location)
                finalPos.append(location)
            else:
                # Check for collision using KDTree
                dist, _ = tree.query(location)
                if dist <= radius:
                    collided = True
                    finalLen[angleCtr] = np.linalg.norm(ctr - location)
                    finalPos.append(location)
        
    return finalLen, finalPos

### Define a function to generate a random point not within any circles
def random_point(bounds, o_centers, radius):
    while True:
        point = np.array([rng.uniform()*bounds[0], rng.uniform()*bounds[1]])
        if all(np.linalg.norm(point - o_center) > radius for o_center in o_centers):
            return point


# ### Example of 6 um, 0x


# Create the 'Blues' colormap
blues_cmap = plt.get_cmap('Blues_r')

# Define the range to exclude the white/light colors (typically at the higher end)
# Let's exclude the top 10% of the colormap to avoid the lightest colors
n_colors = blues_cmap.N

# Get the colors, excluding the bottom 20% and the top 30%
colors = blues_cmap(np.linspace(0.2, 0.7, n_colors))

# Create a new colormap from the modified colors
custom_cmap = LinearSegmentedColormap.from_list('CustomBlues', colors)


# In[ ]:


### Chord length calculation in representative region 1 (C = 6 um; D = 0) [Figure 4a left]

# Random shift
maxshift = 0
# Distance between pillars
delta = 16
# Pillar size
R = 5  # Radius

cc = -1
allpos = []
allpos0 = []
# Number of pillars
Npils = len(np.arange(0, 400, delta))
rng = np.random.default_rng(12345)

allshifts = rng.random((Npils, Npils, 2)) - 0.5
cx = -1
for x in np.arange(0, 400, delta):
    cx = cx + 1
    cy = -1
    for y in np.arange(0, 400, delta):
        cy = cy + 1
        allpos0.append([x, y])
        cc = cc + 1
        rng = np.random.default_rng(cc)
        pos = np.array([x, y]) + allshifts[cx, cy] * maxshift
        allpos.append(pos)  # store the locations of circles

x = [arr[0] for arr in allpos]
y = [arr[1] for arr in allpos]

xmin = min(x)
xmax = max(x)

ymin = min(y)
ymax = max(y)

bounds = [xmax+R, ymax+R]


# Generate N number random points not within any circles
num_centers = 100
centers = [random_point(bounds, allpos, R) for _ in range(num_centers)]

numSamples = 100

# Create KDTree for fast collision detection
tree = KDTree(allpos)

fig, ax = plt.subplots(figsize=(10,10))

chords = []

all_maxl = []
all_minl = []

# Calculate and plot chords for each random point
for center in centers:
    startPos = [center] * numSamples
    results = ChordLength(center, tree, R, bounds)
    chords.append(results[0])
    
    #chord_ls = [np.linalg.norm(np.array(end)-np.array(center)) for end in results[1]] #
    
    all_maxl.append(max(results[0]))
    all_minl.append(min(results[0]))

norm = Normalize(vmin = min(all_minl), vmax = max(all_maxl))

for center in centers:
    startPos = [center] * numSamples
    results = ChordLength(center, tree, R, bounds)
    
    colors = [custom_cmap(norm(length)) for length in results[0]]
    
    for start, end, color in zip(startPos, results[1], colors):
        plot = plt.plot([start[0], end[0]], [start[1], end[1]], zorder=1, color=color)

    plt.scatter(center[0],center[1], c='xkcd:gold',s=20,zorder=2)
        
# Draw circles
mpl.rcParams['font.family'] = 'Arial'
circles = [plt.Circle((xi, yi), radius=5, linewidth=0) for xi, yi in zip(x, y)]
c = mpl.collections.PatchCollection(circles, color='xkcd:grey', alpha=0.3)

ax.set_xlim(xmin - 2*R, xmax + 2*R)
ax.set_ylim(ymin - 2*R, ymax + 2*R)
ax.add_collection(c)
ax.set_aspect(1)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])  # dummy mappable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)
cbar = plt.colorbar(sm, cax=cax, shrink=0.75)
cbar.set_label('Chord length (µm)', fontsize=20)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.show()


# In[ ]:


### Chord length calculation in representative region 1 (C = 1.3 um; D = 3)

# Random shift
maxshift = 3.9
# Distance between pillars
delta = 11.3
# Pillar size
R = 5  # Radius

cc = -1
allpos = []
allpos0 = []
# Number of pillars
Npils = len(np.arange(0, 400, delta))
rng = np.random.default_rng(12345)

allshifts = rng.random((Npils, Npils, 2)) - 0.5
cx = -1
for x in np.arange(0, 400, delta):
    cx = cx + 1
    cy = -1
    for y in np.arange(0, 400, delta):
        cy = cy + 1
        allpos0.append([x, y])
        cc = cc + 1
        rng = np.random.default_rng(cc)
        pos = np.array([x, y]) + allshifts[cx, cy] * maxshift
        allpos.append(pos)  # store the locations of circles

x = [arr[0] for arr in allpos]
y = [arr[1] for arr in allpos]

xmin = min(x)
xmax = max(x)

ymin = min(y)
ymax = max(y)

bounds = [xmax+R, ymax+R]


# Generate N number random points not within any circles
num_centers = 100
centers = [random_point(bounds, allpos, R) for _ in range(num_centers)]

numSamples = 100

# Create KDTree for fast collision detection
tree = KDTree(allpos)

fig, ax = plt.subplots(figsize=(10,10))

# Create a 
chords = []

all_maxl = []
all_minl = []

# Calculate and plot chords for each random point
for center in centers:
    startPos = [center] * numSamples
    results = ChordLength(center, tree, R, bounds)
    chords.append(results[0])
    
    #chord_ls = [np.linalg.norm(np.array(end)-np.array(center)) for end in results[1]] #

    all_maxl.append(max(results[0]))
    all_minl.append(min(results[0]))

norm = Normalize(vmin = min(all_minl), vmax = max(all_maxl))

for center in centers:
    startPos = [center] * numSamples
    results = ChordLength(center, tree, R, bounds)
    
    colors = [custom_cmap(norm(length)) for length in results[0]]
    
    for start, end, color in zip(startPos, results[1], colors):
        plot = plt.plot([start[0], end[0]], [start[1], end[1]], zorder=1, color=color)
    plt.scatter(center[0],center[1], c='xkcd:gold',s=10,zorder=2)
        
# Draw circles
mpl.rcParams['font.family'] = 'Arial'
circles = [plt.Circle((xi, yi), radius=5, linewidth=0) for xi, yi in zip(x, y)]
c = mpl.collections.PatchCollection(circles, color='xkcd:black', alpha=0.1)

ax.set_xlim(xmin - 2*R, xmax + 2*R)
ax.set_ylim(ymin - 2*R, ymax + 2*R)
ax.add_collection(c)
ax.set_aspect(1)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])  # dummy mappable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)
cbar = plt.colorbar(sm, cax=cax, shrink=0.75)
cbar.set_label('Chord length (µm)', fontsize=20)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.show()


# In[ ]:


# Zoomed-in: Calculate and plot chords for one random point

fig, ax = plt.subplots(figsize=(10,10))
center = np.array([390.94199039,  21.60377202])#[356.85676226, 194.08379509])
startPos = [center] * numSamples
results = ChordLength(center, tree, R,bounds)

norm = Normalize(vmin = min(results[0]), vmax = max(results[0]))
colors = [custom_cmap(norm(length)) for length in results[0]]

for start, end, color in zip(startPos, results[1],colors):
    plt.plot([start[0], end[0]], [start[1], end[1]], zorder=1, color=color)
plt.scatter(center[0],center[1], c='xkcd:gold',s=50,zorder=2, label='origin')
        
# Draw circles
mpl.rcParams['font.family'] = 'Arial'
circles = [plt.Circle((xi, yi), radius=5, linewidth=0) for xi, yi in zip(x, y)]
c = mpl.collections.PatchCollection(circles, color='xkcd:black',alpha=0.1)

ax.set_xlim(center[0]-2.5*R,center[0]+2.5*R)
ax.set_ylim(center[1]-2.5*R,center[1]+2.5*R)
ax.add_collection(c)
ax.set_aspect(1)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.legend(fontsize=26)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])  # dummy mappable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)
cbar = plt.colorbar(sm, cax=cax, shrink=0.75)
cbar.set_label('Chord length (µm)', fontsize=20)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.show()


# In[ ]:


### Simulate the chords for all regions (100 origins; each origin has 100 chords) [Supplementary Figure 6]
# Distance between pillars
deltas = [16, 12.6, 11.3]
# Pillar size
R = 5  # Radius

# Random shift
maxshifts = []

for delta in deltas:
    reg = 0
    one = round(delta - 2*R,2)
    two = round(one * 2,2)
    thr = round(one * 3,2)
    l = [reg,one,two,thr]
    maxshifts.append(l)

#Number of centers
num_centers = 100

#Store positions and centers
positions = []
cen = []
bounds = []
for j, delta in enumerate(deltas):
    for i, maxshift in enumerate(maxshifts[j]):

        cc = -1
        allpos = []
        allpos0 = []
        # Number of pillars
        Npils = len(np.arange(0, 400, delta))
        random.seed(42)
        rng = np.random.default_rng(12345)

        allshifts = rng.random((Npils, Npils, 2)) - 0.5
        cx = -1
        for x in np.arange(0, 400, delta):
            cx = cx + 1
            cy = -1
            for y in np.arange(0, 400, delta):
                cy = cy + 1
                allpos0.append([x, y])
                cc = cc + 1
                rng = np.random.default_rng(cc)
                pos = np.array([x, y]) + allshifts[cx, cy] * maxshift
                allpos.append(pos)  # store the locations of circles
        
        x = [arr[0] for arr in allpos]
        y = [arr[1] for arr in allpos]

        xmin = min(x)
        xmax = max(x)

        ymin = min(y)
        ymax = max(y)
        
        boundary = [xmax+R, ymax+R]
        
        bounds.append(boundary)
        
        # Generate N number random points not within any circles
        centers = [random_point(boundary, allpos, R) for _ in range(num_centers)]

        positions.append(allpos)
        cen.append(centers)

mpl.rcParams['font.family'] = 'Arial'
rows = 3
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(26,18))
count = 0 

chords_all = []
all_handles = []
all_labels = []

name = ['C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

for i in range(len(positions)):

    numSamples = 100

    # Create KDTree for fast collision detection
    tree = KDTree(positions[i])
    
    chords = []
    
    all_maxl = []
    all_minl = []
    
    count += 1
    ax = plt.subplot(rows, cols, count)
    
    # Calculate and plot chords for each random point
    for center in cen[i]:
        startPos = [center] * numSamples
        results = ChordLength(center, tree, R, bounds[i])
        chords.extend(results[0])
        
        all_maxl.append(max(results[0]))
        all_minl.append(min(results[0]))
        
    norm = Normalize(vmin = min(all_minl), vmax = max(all_maxl))
        
    for center in cen[i]:
        startPos = [center] * numSamples
        results = ChordLength(center, tree, R, bounds[i])
        
        colors = [custom_cmap(norm(length)) for length in results[0]]
        
        for start, end, color in zip(startPos, results[1], colors):
            plt.plot([start[0], end[0]], [start[1], end[1]], zorder=1,color=color)
        plt.scatter(center[0],center[1], c='xkcd:gold',s=15,zorder=2)
    
    chords_all.append(chords)
    
    # Draw circles
    x = [arr[0] for arr in positions[i]]
    y = [arr[1] for arr in positions[i]]
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    
    mpl.rcParams['font.family'] = 'Arial'
    circles = [plt.Circle((xi, yi), radius=5, linewidth=0) for xi, yi in zip(x, y)]
    c = mpl.collections.PatchCollection(circles, color='xkcd:black',alpha=0.1)

    ax.set_xlim(xmin - 2*R, xmax + 2*R)
    ax.set_ylim(ymin - 2*R, ymax + 2*R)
    ax.set_title(f'{name[i]}',fontsize=26)
    ax.add_collection(c)
    ax.set_aspect(1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # dummy mappable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(sm, cax=cax, shrink=0.75)
    cbar.set_label('Chord length (µm)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

plt.tight_layout() 
plt.savefig('suppfig8_chords_all_100x100.png', transparent=True, dpi=300)  
plt.savefig('suppfig8_chords_all_100x100.pdf', transparent=True, dpi=300) 
plt.close()

