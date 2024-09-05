#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This script is written for computing the chord lengths & run lengths [Figure 4] - Haibei Zhang 
import os
import glob,csv
import math
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

# In[ ]:


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

# In[ ]:


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
plt.savefig('fig4a_left_chord_6um0x.png',transparent=True,dpi=300)
plt.savefig('fig4a_left_chord_6um0x.pdf',transparent=True,dpi=300)
plt.close()


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
plt.savefig('fig4a_center_chord_1um3x.png',transparent=True,dpi=300)
plt.savefig('fig4a_center_chord_1um3x.pdf',transparent=True,dpi=300)
plt.close()


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
plt.savefig('fig4a_right_chord_1um3x_enlarged.png',transparent=True,dpi=300)
plt.close()


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
plt.savefig('suppfig6_chords_all_100x100.png', transparent=True, dpi=300)  
plt.savefig('suppfig6_chords_all_100x100.pdf', transparent=True, dpi=300) 
plt.close()


# In[ ]:


### Plot the KDEs of chord length (100 origins; each origin has 100 chords) [Supplementary Figure 7a]
# Create a dictionary for simulated chords 
dic = {}
for i, chords in enumerate(chords_all):
    cl = [round(elem, 2) for elem in chords_all[i]]
    label = 'CL in ' + name[i]
    dic[label] = cl

CL_all = pd.DataFrame(dic)

fig, ax = plt.subplots(figsize=(8, 8))
mpl.rcParams['font.family'] = 'Arial'
sns.set_style('white')
sns.set_style('ticks')
colors = ['xkcd:black','xkcd:grey','xkcd:greyish','xkcd:light grey',
      'xkcd:darkblue', 'xkcd:sea blue', 'xkcd:mid blue', 'xkcd:baby blue', 
         'xkcd:rust', 'xkcd:orange', 'xkcd:yellowish orange','xkcd:dull yellow']

mean_peak = []

for i in range(len(colors)):
    fig_temp, ax_temp = plt.subplots()  # Create a temporary figure for each plot
    
    plot = sns.kdeplot(data=CL_all, x=CL_all[CL_all.columns[i]], color=colors[i], ax=ax_temp, label=CL_all.columns[i], linewidth=2)
    
    x = plot.lines[0].get_xdata()  # Get the x data of the distribution
    y = plot.lines[0].get_ydata()  # Get the y data of the distribution
    
    maxid = np.argmax(y)  # The id of the peak (maximum of y data)
    
    peak_x = x[maxid]
    peak_y = y[maxid]
    
    #print(f"Peak for {CL_all.columns[i]}: {peak_x}")

    mean_peak.append([peak_x, peak_y])
    
    plt.close(fig_temp)  # Close the temporary figure
    
    # Re-plot the KDE on the main figure
    sns.kdeplot(data=CL_all, x=CL_all[CL_all.columns[i]], color=colors[i], ax=ax, label=CL_all.columns[i], linewidth=2,zorder=1)
    ax.scatter(peak_x, peak_y, color='k',marker='*',s=100, zorder=2)  # Plot a scatter point at the peak on the main figure
plt.xlim(0.0001, 80)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Chord Lengths, CL (µm)', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.legend(fontsize=20)
sns.despine()
plt.tight_layout()
plt.savefig('supplfig7a_CL_all_100x100.png',transparent = True, dpi = 300)
plt.close()


# In[ ]:


### Plot the KDEs of chord length (200 origins; each origin has 100 chords) [Supplementary Figure 7b]

## Simulate the chords for all regions (200 origins; each origin has 100 chords) 
#Number of centers
num_centers = 200

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


chords_all_1 = []

for i in range(len(positions)):

    numSamples = 100

    # Create KDTree for fast collision detection
    tree = KDTree(positions[i])
    
    chords = []
    
    count += 1
    #ax = plt.subplot(rows, cols, count)
    
    # Calculate and plot chords for each random point
    for center in cen[i]:
        startPos = [center] * numSamples
        results = ChordLength(center, tree, R, bounds[i])
        chords.extend(results[0])

    chords_all_1.append(chords)
    
### Create a dictionary for simulated chords 
dic1 = {}
for i, chords in enumerate(chords_all_1):
    cl = [round(elem, 2) for elem in chords_all_1[i]]
    label = 'CL in ' + name[i]
    dic1[label] = cl

CL_all_1 = pd.DataFrame(dic1)

fig, ax = plt.subplots(figsize=(8, 8))
mpl.rcParams['font.family'] = 'Arial'
sns.set_style('white')
sns.set_style('ticks')
colors = ['xkcd:black','xkcd:grey','xkcd:greyish','xkcd:light grey',
      'xkcd:darkblue', 'xkcd:sea blue', 'xkcd:mid blue', 'xkcd:baby blue', 
         'xkcd:rust', 'xkcd:orange', 'xkcd:yellowish orange','xkcd:dull yellow']

mean_peak = []

for i in range(len(colors)):
    fig_temp, ax_temp = plt.subplots()  # Create a temporary figure for each plot
    
    plot = sns.kdeplot(data=CL_all_1, x=CL_all_1[CL_all_1.columns[i]], color=colors[i], ax=ax_temp, label=CL_all_1.columns[i], linewidth=2)
    
    x = plot.lines[0].get_xdata()  # Get the x data of the distribution
    y = plot.lines[0].get_ydata()  # Get the y data of the distribution
    
    maxid = np.argmax(y)  # The id of the peak (maximum of y data)
    
    peak_x = x[maxid]
    peak_y = y[maxid]
    
    mean_peak.append([peak_x, peak_y])
    
    plt.close(fig_temp)  # Close the temporary figure
    
    # Re-plot the KDE on the main figure
    sns.kdeplot(data=CL_all_1, x=CL_all_1[CL_all_1.columns[i]], color=colors[i], ax=ax, label=CL_all_1.columns[i], linewidth=2,zorder=1)
    ax.scatter(peak_x, peak_y, color='k',marker='*',s=100, zorder=2)  # Plot a scatter point at the peak on the main figure
plt.xlim(0.0001, 80)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Chord Lengths, CL (µm)', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.legend(fontsize=20)
sns.despine()
plt.tight_layout()
plt.savefig('suppfig7b_CL_all_200x100.png',transparent = True, dpi = 300)
plt.close()


# In[ ]:


### Compute run lengths 
# Define a function to calculate the velocity 
def calculate_velocity(x_positions, y_positions, time_positions,interval:int, fps):
    #Initialize the velocity arrays with NaNs
    vx = np.full(len(x_positions),np.nan)
    vy = np.full(len(y_positions),np.nan)
    
    # Calculate the differences in positions and time
    
    valid_indices = np.arange(interval, len(x_positions))
    dx = np.array(x_positions)[valid_indices] - np.array(x_positions)[:-interval]
    dy = np.array(y_positions)[valid_indices] - np.array(y_positions)[:-interval]
    dt = (np.array(time_positions)[valid_indices] - np.array(time_positions)[:-interval]) / fps 
    
    # Calculate velocity components vx and vy
    vx[valid_indices] = dx / dt
    vy[valid_indices] = dy / dt
    
    # Return velocity vectors as a list of arrays
    velocity_vectors = [np.array([vx[i], vy[i]]) for i in range(len(vx))]
    
    return velocity_vectors

# Define a function to calculate the angle
def dtheta(list_of_vels):
    dtheta = [np.nan]
    for i in range(1, len(list_of_vels)-1):
        vt = list_of_vels[i]
        v_dt = list_of_vels[i+1]
        v_dot = np.dot(vt,v_dt)
        v_cross = np.cross(vt,v_dt)
        theta = abs(np.arctan2(v_cross,v_dot))
        #deg = np.rad2deg(abs(np.arctan2(v_cross,v_dot)))
        dtheta.append(theta)
    dtheta.append(np.nan)
    return dtheta 

# Define a function to compute the run lengths (Assume the dataframe has the rescaled x,y position data, correpsonding speed, angle and time data:)
def run_lengths(speed_thresh, angle_thresh, df: pd.DataFrame):
    run_ls = []
    first = []
    last = []
    start_point = None

    for idx, row in df.iterrows():
        x = row['rescaled_pos'][0]
        y = row['rescaled_pos'][1]
        #print(rescaled_pos[0].dtype)      
        speed = row['speed']
        angle = row['angle']
      
        if pd.notna(speed) and  pd.notna(angle) and speed >= speed_thresh and angle <= angle_thresh:
            if start_point is None:
                # Start of a new run
                start_point = (x, y)
            end_point = (x, y)
        else:
            if start_point is not None:
                # End of the current run
                first.append(start_point)
                last.append(end_point)
                distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
                if distance != 0:     
                    run_ls.append(distance)
                
                start_point = None  # Reset start_point after ending the run

    # If a run was ongoing at the end of the loop, add the final segment
    if start_point is not None:
        first.append(start_point)
        last.append(end_point)
        distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        run_ls.append(distance)
    
    return run_ls

# Retrieve the data
upperdir = '/'.join(os.getcwd().split('/')[:-1]) #define upper directory
lowerdir = '/sorted_tracks/' #define lower directory where all the data is located
files = glob.glob(upperdir + lowerdir + '*.csv') #grab all the files in their pathways 
files = sorted(files) #sort the files based on the naming
len(files)

# Set up a dictionary to store the info from the files 
datadic = {} #set up a data directory
dev = [] #number of device
pil= [] #number of confinement
dis = [] #number of disorders
rep = [] #number of repititions 
pixel_to_micron = 0.656 #pixel to micron (from Miles)
fps = 20 #frames per second
interval = [1,2]

for file in files:
    
    curr_data = pd.read_csv(file)
    data = curr_data[['TRACK_ID','POSITION_X','POSITION_Y','POSITION_T']]
    grouped_df = data.groupby('TRACK_ID')
    # Filter out data with less spots
    minimum_spots = 160  # Replace this with a desired "minimum_spots" value
    filtered_df = grouped_df.filter(lambda group: len(group) >= minimum_spots)
    # Reindex the track IDs starting from 1
    filtered_df.reset_index(drop=True, inplace=True) #reindex the column starting from 0
    df = filtered_df.groupby('TRACK_ID', group_keys=True, as_index=False).apply(lambda x: x) #group the dataframe by TRACK_ID
    df.reset_index(level=0, inplace=True) #move the first level to a separate column
    df.rename(columns={'level_0': 'ID'}, inplace=True) #rename new level  
    track_names = df['ID'].unique().tolist()
    #print(df)
    
    device = file.split('-')[-5] #retrieve the strains used in each microfluidic device 
    pillar = file.split('-')[-4] #retrieve the specific confinement 
    disorder = file.split('-')[-3] #retrieve the specific disorder degree
    #repetition = file.split('-')[-1][:4] #the first four letter
    if device not in datadic:
        dev.extend([device])
        datadic[device] = {} 
    if pillar not in datadic[device]:
        if pillar not in pil:
            pil.extend([pillar])
        datadic[device][pillar] = {}
    if disorder not in datadic[device][pillar]:
        if disorder not in dis:
            dis.extend([disorder])
        datadic[device][pillar][disorder] = {}
        datadic[device][pillar][disorder]['time'] = {}
        datadic[device][pillar][disorder]['rescaled_time'] = {} #Create a list to store all 'rescaled' time info
        datadic[device][pillar][disorder]['pos'] = {}
        datadic[device][pillar][disorder]['rescaled_pos'] = {} #Create a list to store all 'rescaled' x and y info
        #datadic[device][pillar][disorder]['MSD'] = [[]]
        datadic[device][pillar][disorder]['speed'] = {}
        datadic[device][pillar][disorder]['angle'] = {}

    for dt in interval:
        if dt not in datadic[device][pillar][disorder]['time']:
            datadic[device][pillar][disorder]['time'][dt] = []
        if dt not in datadic[device][pillar][disorder]['rescaled_time']:
            datadic[device][pillar][disorder]['rescaled_time'][dt] = []
        if dt not in datadic[device][pillar][disorder]['pos']:
            datadic[device][pillar][disorder]['pos'][dt] = []
        if dt not in datadic[device][pillar][disorder]['rescaled_pos']:
            datadic[device][pillar][disorder]['rescaled_pos'][dt] = []    
        if dt not in datadic[device][pillar][disorder]['speed']:
            datadic[device][pillar][disorder]['speed'][dt] = []
        if dt not in datadic[device][pillar][disorder]['angle']:
            datadic[device][pillar][disorder]['angle'][dt] = []
        

        for i in range(len(track_names)):
            time = np.array(df[df['ID']==track_names[i]]['POSITION_T'].tolist())
            datadic[device][pillar][disorder]['time'][dt].append(time)
            datadic[device][pillar][disorder]['rescaled_time'][dt].append([time[i]/fps for i in range(len(time))]) #rescale the time as well based on fps
            
            x_pos = np.array(df[df['ID']==track_names[i]]['POSITION_X'].tolist())
            y_pos = np.array(df[df['ID']==track_names[i]]['POSITION_Y'].tolist())
            x_pos_rescaled = x_pos*pixel_to_micron
            y_pos_rescaled = y_pos*pixel_to_micron
            x_start = x_pos[0]
            y_start = y_pos[0] 
            datadic[device][pillar][disorder]['pos'][dt].append([(x_pos[t]-x_start,y_pos[t]-y_start) for t in range(len(x_pos))])
            datadic[device][pillar][disorder]['rescaled_pos'][dt].append([(x_pos_rescaled[t], y_pos_rescaled[t]) for t in range(len(x_pos))])

            velocities = calculate_velocity(x_pos_rescaled, y_pos_rescaled, time, dt, fps)

            angles = dtheta(velocities)

            datadic[device][pillar][disorder]['angle'][dt].append(angles)

            speeds = [np.nan]*len(x_pos)

            for j in range(dt, len(x_pos)):
                speed = np.sqrt(pixel_to_micron*(x_pos[j]-x_pos[j-dt])**2+pixel_to_micron*(y_pos[j]-y_pos[j-dt])**2)/(dt/fps)

                speeds[j] = speed

            datadic[device][pillar][disorder]['speed'][dt].append(speeds)
            
    
# Test whether the trajecories are stored properly           
#     if device == 'dev1' and pillar == 'pil0' and disorder == 'dis0':
#          for i in range(len(datadic[device][pillar][disorder]['rescaled_pos'])):
#             path = np.array(datadic[device][pillar][disorder]['rescaled_pos'][i])
#             plt.plot(*path.T) # ".T" attribute is used to transpose the array, swapping its rows and columns. # "*" unpacks the rows of the transposed array
#             plt.xlabel('µm')
#             plt.ylabel('µm')
    
#     plt.show()

# Collect the data for all angles and speeds and store it in a DataFrame
v_theta = []
for dt in interval:
    for dev in list(datadic):
        for pil in list(datadic[dev]):
            for dis in list(datadic[dev][pil]):
                v_theta.extend([(dt, dev, pil, dis, pos, speed, angle) for pos, speed, angle 
                                in zip(datadic[dev][pil][dis]['rescaled_pos'][dt], 
                                       datadic[dev][pil][dis]['speed'][dt], datadic[dev][pil][dis]['angle'][dt])])

df = pd.DataFrame(v_theta, columns=['dt', 'strain', 'confinement', 'disorder', 'p', 's', 'a'])

pillar_mapping = {'pil0': 'Unc', 'pil1': 'C = 6 µm', 'pil2': 'C = 2.6 µm', 'pil3': 'C = 1.3 µm'}
disorder_mapping = {'dis0': 'no pillar', 'dis1': 'D = 0', 'dis2': 'D = 1', 
                    'dis3': 'D = 2', 'dis4':'D = 3'}

# Replace the category names in the DataFrame
df['confinement'] = df['confinement'].replace(pillar_mapping)
df['disorder'] = df['disorder'].replace(disorder_mapping) 

# Merge the two columns to one 'condition' column
df['condition'] = df['confinement'] + '; ' + df['disorder']
#df.drop(['confinement', 'disorder'], axis=1, inplace=True)
sub_dfs = {dt: group for dt, group in df.groupby('dt')}

df_exploded = sub_dfs[1].explode(list('psa'))
df_exploded['p'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded['s'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded['a'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded = df_exploded.rename(columns={'p':'rescaled_pos', 's':'speed', 'a':'angle'})
df_exploded['condition'] = df_exploded['condition'].astype('category')

average_velocity = df_exploded.groupby(['condition'])['speed'].mean().reset_index()
avg_vel = average_velocity.rename(columns={'speed':'Mean_Speed'})
mean_speed_unconfined = avg_vel.iloc[-1,-1]

half_mean_speed = round(mean_speed_unconfined/2,3)
angle_thresh = round(np.pi/3,3)

def convert_to_numeric(pos):
    if isinstance(pos, list):
        return [pd.to_numeric(coord, errors='coerce') for coord in pos]
    return pos

def flatten(matrix):
    return [item for row in matrix for item in row]

df_exploded['particle_id'] = df_exploded.index
gp_dfs = df_exploded.groupby(['condition', 'particle_id'])
run_ls = {}
for (condition, particle_id), gp in gp_dfs:
    gp['rescaled_pos'] = gp['rescaled_pos'].apply(convert_to_numeric)
    gp['speed'] = gp['speed'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    gp['angle'] = gp['angle'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    run_l = run_lengths(half_mean_speed, angle_thresh, gp)
    
    if condition not in run_ls:
        run_ls[condition] = [run_l]
    else: 
        run_ls[condition].append(run_l)

for condition in run_ls:
    run_ls[condition] = flatten(run_ls[condition])
    
# Create a list of dictionaries to hold the dataframe rows
data = []
for condition, lengths in run_ls.items():
    for length in lengths:
        data.append({'condition': condition, 'run_length': length})

# Convert the list of dictionaries into a pandas dataframe
df_run_lengths = pd.DataFrame(data)


# In[ ]:


### Combine two data frames to one for comparison [chord length dataframe + run length dataframe]
df_run_lengths = df_run_lengths.rename(columns={"run_length": "length"})

CL_all_long = CL_all.melt(var_name='condition', value_name='chord_length')

CL_all_long['condition'] = CL_all_long['condition'].str.replace('CL in ', '')

CL_all_long = CL_all_long.rename(columns={"chord_length": "length"})

# Add an indicator column
CL_all_long['type'] = 'Chord Length'

# Add an indicator column to df_run_lengths
df_run_lengths['type'] = 'Run Length'

# Combine the dataframes
combined_cl_rl = pd.concat([CL_all_long,df_run_lengths], ignore_index=True)


# In[ ]:


# Combine box plots with violin plots (https://stackoverflow.com/questions/71925775/showing-both-boxplots-when-using-split-in-seaborn-violinplots)

# Filter for specific conditions
fig, ax = plt.subplots(figsize=(18, 4))

conditions_to_compare = ['Unc; no pillar', 'C = 6 µm; D = 0', 'C = 2.6 µm; D = 0', 'C = 1.3 µm; D = 0', 'C = 6 µm; D = 3', 'C = 2.6 µm; D = 3', 'C = 1.3 µm; D = 3']
filtered_df = combined_cl_rl[combined_cl_rl['condition'].isin(conditions_to_compare)]
sns.set_theme(style="ticks")

# Plot the split violin plot
sns.violinplot(x='condition', y='length', hue='type', data=filtered_df, split=True,
               fill=True,inner=None, palette={"xkcd:azure", "xkcd:darkish blue"})

sns.boxplot(ax=ax,data=filtered_df, x='condition', y='length', hue='type', palette=['xkcd:light grey','white'], width=0.12, showfliers = False)#, showmeans=True, meanprops={'marker':'*','markerfacecolor':'xkcd:lemon','markeredgecolor':'xkcd:green','markersize':'8'},dodge=True, boxprops={'zorder': 2})

plt.ylim(0,60)
plt.xticks(ticks=range(len(conditions_to_compare)), labels=['6 µm', '6 µm', '2.6 µm', '2.6 µm',  '1.3 µm', '1.3 µm', 'Unc'], fontsize=22)
                                                           
plt.yticks(fontsize=18)
ax.set_xlabel('')
ax.set_ylabel('Length (µm)',fontsize=22)
plt.tight_layout()
plt.savefig('fig4b_chord_run_dist.png',transparent=True,dpi=300)
plt.close()


# In[ ]:


#### Display the KDEs of run lenths for all regions [Supplementary figure 8]
mpl.rcParams['font.family'] = 'Arial'
gp = df_run_lengths.groupby('condition')

categories = ['Unc; no pillar', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

labels_rl = ['Unc', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

colors = ['xkcd:grey'] + sns.color_palette('Blues_r')[:4] +[
    'xkcd:dark orange', 'xkcd:orange', 'xkcd:peach','xkcd:golden']+[
    'xkcd:dark green', 'xkcd:grass', 'xkcd:jade','xkcd:slime green']

fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.set_style('white')
sns.set_style('ticks')

ax = plt.subplot(1,3,1)

categories = ['Unc; no pillar', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3']

for j, cato in enumerate(categories):
    
    sns.kdeplot(data=gp.get_group(cato),x='length',ax=ax,label = labels_rl[j],color=colors[j],linewidth=2)

plt.xlim(0,30)
plt.ylim(0,0.18)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,0.21,0.03),fontsize=20)
plt.xlabel('Run Lengths, RL (µm)',fontsize=24)
plt.ylabel('Density',fontsize=24)
plt.legend(loc='upper center',fontsize=16)
sns.despine()

ax = plt.subplot(1,3,2)

categories = ['Unc; no pillar', 'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3']

labels_rl = ['Unc', 'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3']

colors = ['xkcd:grey']+['xkcd:dark orange', 'xkcd:orange', 'xkcd:peach','xkcd:golden']

for j, cato in enumerate(categories):
    
    sns.kdeplot(data=gp.get_group(cato),x='length',ax=ax,label = labels_rl[j],color=colors[j],linewidth=2)
    
plt.xlim(0,30)
plt.ylim(0,0.18)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,0.21,0.03),fontsize=20)
plt.xlabel('Run Lengths, RL (µm)',fontsize=24)
plt.ylabel('Density',fontsize=24)
plt.legend(loc='upper center',fontsize=16)
sns.despine()

ax = plt.subplot(1,3,3)

categories = ['Unc; no pillar', 'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

labels_rl = ['Unc', 'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

colors = ['xkcd:grey']+['xkcd:dark green', 'xkcd:grass', 'xkcd:jade','xkcd:slime green']

for j, cato in enumerate(categories):
    
    sns.kdeplot(data=gp.get_group(cato),x='length',ax=ax,label = labels_rl[j],color=colors[j],linewidth=2)
    
plt.xlim(0,30)
plt.ylim(0,0.18)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,0.21,0.03),fontsize=20)
plt.xlabel('Run Lengths, RL (µm)',fontsize=24)
plt.ylabel('Density',fontsize=24)
plt.legend(loc='upper center',fontsize=16)
sns.despine()
plt.tight_layout()
plt.savefig('supplefig8.png',transparent=True,dpi=300)
plt.close()


# In[ ]:


##### Compute the ratio of mean run lengths (with respect to the mean run length in the unconfined) [Fig 4c]
mean_rl = df_run_lengths.groupby(['condition'])['length'].mean().tolist()
ratios = []
for i in mean_rl:
    ratio = round(i/mean_rl[-1],3)
    ratios.append(ratio)
reshaped_ratios = [ratios[i:i+4] for i in range(0, (len(ratios)-1), 4)]
transposed_r= [list(i) for i in zip(*reshaped_ratios)]
transposed_r

mpl.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(8,5))
t = [1,2,3]
plt.plot(t,transposed_r[0][::-1],linestyle='dashed',color='xkcd:gray')
plt.scatter(t,transposed_r[0][::-1],s=50,label='D = 0',color='xkcd:gray')
plt.plot(t,transposed_r[1][::-1],linestyle='dashed',color='purple')
plt.scatter(t,transposed_r[1][::-1],s=50,color='purple',label='D = 1')
plt.plot(t,transposed_r[2][::-1],linestyle='dashed',color='xkcd:azure')
plt.scatter(t,transposed_r[2][::-1],s=50,color='xkcd:azure',label='D = 2')
plt.plot(t,transposed_r[3][::-1],linestyle='dashed',color='xkcd:orange')
plt.scatter(t,transposed_r[3][::-1],s=50,label='D = 3',color='xkcd:orange')
plt.legend(fontsize=20)
plt.xlim(0.6,3.3)
plt.xticks(np.arange(1,4,1),['6 µm','2.6 µm','1.3 µm'],fontsize=20)
plt.xlabel('C',fontsize=20)
plt.ylabel(r'$ε_{unc}$ = $\frac{l_{r_{C}}}{l_{r_{unc}}}$',fontsize=26)
plt.yticks(fontsize=20)
plt.ylim(0.30,0.60)
plt.tight_layout()
plt.savefig('fig4c_runlengthratio.png',transparent=True,dpi=300)
plt.savefig('fig4c_runlengthratio.pdf',transparent=True,dpi=300)
plt.close()


# In[ ]:


##### Compute the ratio of mean run lengths to mean chord lengths [Fig 4d]
mean_cl = CL_all.mean().tolist()
mean_rl_nounc = mean_rl[:-1]
mean_rl_nounc.reverse()

ratios_1 = []
for ind, i in enumerate(mean_rl_nounc):
    ratio_1 = round(i/mean_cl[ind],3)
    ratios_1.append(ratio_1)
    
reshaped_ratios_1 = [ratios_1[i:i+4] for i in range(0, len(ratios_1), 4)]
reshaped_ratios_1
transposed_r_1= [list(i) for i in zip(*reshaped_ratios_1)]

mpl.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(8,5))
t = [1,2,3]
plt.plot(t,transposed_r_1[0],linestyle='dashed',color='xkcd:grey')
plt.scatter(t,transposed_r_1[0],s=50,label='D = 0',color='xkcd:grey')
plt.plot(t,transposed_r_1[1],linestyle='dashed',color='purple')
plt.scatter(t,transposed_r_1[1],s=50,color='purple',label='D = 1')
plt.plot(t,transposed_r_1[2],linestyle='dashed',color='xkcd:azure')
plt.scatter(t,transposed_r_1[2],s=50,color='xkcd:azure',label='D = 2')
plt.plot(t,transposed_r_1[3],linestyle='dashed',color='xkcd:orange')
plt.scatter(t,transposed_r_1[3],s=50,label='D = 3',color='xkcd:orange')
plt.legend(fontsize=20)
plt.xlim(0.6,3.3)
plt.xticks(np.arange(1,4,1),['6 µm','2.6 µm','1.3 µm'],fontsize=20)
plt.xlabel('C',fontsize=20)
plt.ylabel(r'$ε_{chord}$ = $\frac{l_{run}}{l_{chord}}$',fontsize=26)
plt.yticks(fontsize=20)
plt.ylim(0.32,1.1)
plt.tight_layout()
plt.savefig('fig4d_chord_run_ratio.png',transparent=True,dpi=300)
plt.savefig('fig4d_chord_run_ratio.pdf',transparent=True,dpi=300)
plt.close()

