#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### The script produces sample trajectories shown in Fig 1, all trajectory overlays in Supplementary Fig. 1,
### and Mean-squared Displacement analysis in Fig 4 and Supplementary Fig. 4 & 5
import os
import glob,csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools as it
from collections import Counter
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit


# In[ ]:


upperdir = '/'.join(os.getcwd().split('/')[:-1]) #define upper directory
lowerdir = '/sorted_tracks/'  #define lower directory where all the data is located

files = glob.glob(upperdir + lowerdir + '*.csv') #grab all the files in their pathways 
files = sorted(files) #sort the files based on the naming


# In[ ]:


##### Plot maps for sample trajectories!

#Construct a dictionary for data storage and analysis
datadic = {} #set up a data directory
dev = [] #number of device
pil= [] #number of confinement
dis = [] #number of disorders
rep = [] #number of repititions 
pixel_to_micron = 0.656 #pixel to micron (from Miles)
fps = 20 #frames per second

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
        datadic[device][pillar][disorder]['time'] = [] #Create a list to store all 'rescaled' time info
        datadic[device][pillar][disorder]['rescaled_time'] = [] #Create a list to store all 'rescaled' time info
        datadic[device][pillar][disorder]['pos'] = []
        datadic[device][pillar][disorder]['rescaled_pos'] = [] #Create a list to store all 'rescaled' x and y info
        
    for i in range(len(track_names)):
        time = np.array(df[df['ID']==track_names[i]]['POSITION_T'].tolist())
        
        time_start = time[0]        
       
        datadic[device][pillar][disorder]['time'].append(time)
        
        datadic[device][pillar][disorder]['rescaled_time'].append([(time[i]-time_start)/fps for i in range(len(time))]) #rescale the time as well based on fps
                
        x_pos = np.array(df[df['ID']==track_names[i]]['POSITION_X'].tolist())
        y_pos = np.array(df[df['ID']==track_names[i]]['POSITION_Y'].tolist())
        x_start = x_pos[0]
        y_start = y_pos[0] 
        datadic[device][pillar][disorder]['pos'].append([(x_pos[t]-x_start,y_pos[t]-y_start) for t in range(len(x_pos))])
        datadic[device][pillar][disorder]['rescaled_pos'].append([(pixel_to_micron*x_pos[t], pixel_to_micron*y_pos[t]) for t in range(len(x_pos))]) 
        


# In[ ]:


mpl.rcParams['font.family'] = 'Arial'

# Plot trajectories for 'unconfined' with gradient color 'bone'
device = 'dev1'
pillar = 'pil0'
disorder = 'dis0'
num_trajs = 20

if device in datadic and pillar in datadic[device] and disorder in datadic[device][pillar]:
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-20, 420)
    ax.set_ylim(-20, 420)
    ax.set_box_aspect(1)

    # Create a normalization object with the global min and max time
    norm = mpl.colors.Normalize(vmin=0.05, vmax=40.0)
    
    # Get the total number of trajectories available
    total_trajectories = len(datadic[device][pillar][disorder]['rescaled_pos'])

    random.seed(2020)
    # Randomly select 50 unique indices
    random_indices = random.sample(range(total_trajectories), num_trajs)
    
    for i in random_indices:
        path = np.array(datadic[device][pillar][disorder]['rescaled_pos'][i])
        time = np.array(datadic[device][pillar][disorder]['rescaled_time'][i])
        scatter = ax.scatter(path[:, 0], path[:, 1], c=time, cmap='bone', norm=norm, s=5) # s is the size of the points
    
    plt.xlabel('', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.xticks([])
    plt.yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax,shrink=0.75)
    cbar.set_label('Time (s)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

plt.savefig('fig1c_map1.png',dpi=300, transparent=True)
plt.close()

### Plot trajectories for 'C = 6, D = 0' with gradient color 'copper'
device = 'dev1'
pillar = 'pil1'
disorder = 'dis1'
num_trajs = 20

if device in datadic and pillar in datadic[device] and disorder in datadic[device][pillar]:
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-20, 420)
    ax.set_ylim(-20, 420)
    ax.set_box_aspect(1)

    # Create a normalization object with the global min and max time
    norm = mpl.colors.Normalize(vmin=0.05, vmax=40.0)
    
    # Get the total number of trajectories available
    total_trajectories = len(datadic[device][pillar][disorder]['rescaled_pos'])

    random.seed(2020)
    # Randomly select 50 unique indices
    random_indices = random.sample(range(total_trajectories), num_trajs)
    
    for i in random_indices:
        path = np.array(datadic[device][pillar][disorder]['rescaled_pos'][i])
        time = np.array(datadic[device][pillar][disorder]['rescaled_time'][i])
        scatter = ax.scatter(path[:, 0], path[:, 1], c=time, cmap='copper', norm=norm, s=5) # s is the size of the points
    
    plt.xlabel('', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.xticks([])
    plt.yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax,shrink=0.75)
    cbar.set_label('Time (s)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

plt.savefig('fig1c_map2.png',dpi=300, transparent=True)
plt.close()

### Plot trajectories for 'C = 1.3, D = 0' with gradient color 'winter'

device = 'dev1'
pillar = 'pil3'
disorder = 'dis1'
num_trajs = 20

if device in datadic and pillar in datadic[device] and disorder in datadic[device][pillar]:
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-20, 420)
    ax.set_ylim(-20, 420)
    ax.set_box_aspect(1)

    # Create a normalization object with the global min and max time
    norm = mpl.colors.Normalize(vmin=0.05, vmax=40.0)
    
    # Get the total number of trajectories available
    total_trajectories = len(datadic[device][pillar][disorder]['rescaled_pos'])

    random.seed(2020)
    # Randomly select 50 unique indices
    random_indices = random.sample(range(total_trajectories), num_trajs)
    
    for i in random_indices:
        path = np.array(datadic[device][pillar][disorder]['rescaled_pos'][i])
        time = np.array(datadic[device][pillar][disorder]['rescaled_time'][i])
        scatter = ax.scatter(path[:, 0], path[:, 1], c=time, cmap='winter', norm=norm, s=5) # s is the size of the points
    
    plt.xlabel('', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.xticks([])
    plt.yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax,shrink=0.75)
    cbar.set_label('Time (s)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)   
plt.savefig('fig1c_map3.png',dpi=300, transparent=True)
plt.close()

### Plot trajectories for 'C = 1.3, D = 3' with gradient color 'summer'

device = 'dev1'
pillar = 'pil3'
disorder = 'dis4'
num_trajs = 20

if device in datadic and pillar in datadic[device] and disorder in datadic[device][pillar]:
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-20, 420)
    ax.set_ylim(-20, 420)
    ax.set_box_aspect(1)

    # Create a normalization object with the global min and max time
    norm = mpl.colors.Normalize(vmin=0.05, vmax=40.0)
    
    # Get the total number of trajectories available
    total_trajectories = len(datadic[device][pillar][disorder]['rescaled_pos'])

    random.seed(2020)
    # Randomly select 50 unique indices
    random_indices = random.sample(range(total_trajectories), num_trajs)
    
    for i in random_indices:
        path = np.array(datadic[device][pillar][disorder]['rescaled_pos'][i])
        time = np.array(datadic[device][pillar][disorder]['rescaled_time'][i])
        scatter = ax.scatter(path[:, 0], path[:, 1], c=time, cmap='summer', norm=norm, s=5) # s is the size of the points
    
    plt.xlabel('', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.xticks([])
    plt.yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax,shrink=0.75)
    cbar.set_label('Time (s)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    
plt.savefig('fig1c_map4.png',dpi=300, transparent=True)
plt.close()


# In[ ]:


##### MSD & Whole Trajectory Overlay
#Construct a dictionary for data storage and analysis
datadic = {} #set up a data directory
dev = [] #number of device
pil= [] #number of confinement
dis = [] #number of disorders
rep = [] #number of repititions 
pixel_to_micron = 0.656 #pixel to micron (from Miles)
fps = 20 #frames per second

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
        datadic[device][pillar][disorder]['time'] = []
        datadic[device][pillar][disorder]['rescaled_time'] = [] #Create a list to store all 'rescaled' time info
        datadic[device][pillar][disorder]['pos'] = []
        datadic[device][pillar][disorder]['rescaled_pos'] = [] #Create a list to store all 'rescaled' x and y info
        datadic[device][pillar][disorder]['MSD'] = [[]]
        
    for i in range(len(track_names)):
        time = np.array(df[df['ID']==track_names[i]]['POSITION_T'].tolist())
        time_start = time[0]        
       
        datadic[device][pillar][disorder]['time'].append(time)
        
        datadic[device][pillar][disorder]['rescaled_time'].append([(time[i]-time_start)/fps for i in range(len(time))]) #rescale the time as well based on fps
        x_pos = np.array(df[df['ID']==track_names[i]]['POSITION_X'].tolist())
        y_pos = np.array(df[df['ID']==track_names[i]]['POSITION_Y'].tolist())
        x_start = x_pos[0]
        y_start = y_pos[0] 
        datadic[device][pillar][disorder]['pos'].append([(x_pos[t]-x_start,y_pos[t]-y_start) for t in range(len(x_pos))])
        datadic[device][pillar][disorder]['rescaled_pos'].append([(pixel_to_micron*x_pos[t], pixel_to_micron*y_pos[t]) for t in range(len(x_pos))]) 
    
        if len(time) < 160:
            continue
        if len(time) > len(datadic[device][pillar][disorder]['MSD'][-1]):
            add_on = len(time) - len(datadic[device][pillar][disorder]['MSD'][-1])
            for j in range(add_on):
                datadic[device][pillar][disorder]['MSD'][-1].extend([[]])
                
        #Calcuate MSD based on msd(τ) = <Δr(τ)2> = <[r(t+τ) - r(t)]2>
        r = [np.sqrt(pixel_to_micron*(x-x_start)**2 + pixel_to_micron*(y-y_start)**2) for (x,y) in zip(x_pos,y_pos)]

        for time in range(len(r)):
            for shift in range(len(r)-time):
                #if shift == 1:
                       #print((r[time+shift] - r[time])**2)
                datadic[device][pillar][disorder]['MSD'][-1][shift].extend([(r[time + shift]-r[time])**2])       
        
        datadic[device][pillar][disorder]['MSD'].extend([[]])
        
#     Test whether the trajecories are stored properly           
#     if device == 'dev1' and pillar == 'pil0' and disorder == 'dis0':
#          for i in range(len(datadic[device][pillar][disorder]['rescaled_pos'])):
#             path = np.array(datadic[device][pillar][disorder]['rescaled_pos'][i])
#             plt.plot(*path.T) # ".T" attribute is used to transpose the array, swapping its rows and columns. # "*" unpacks the rows of the transposed array
#             plt.xlabel('µm')
#             plt.ylabel('µm')
    
#     plt.show() 

### Plot track overlays (Supplementary Figure 1)
mpl.rcParams['font.family'] = 'Arial'
rows = 4
cols = 4
plt.subplots_adjust(hspace=0.6)
plt.figure(figsize=(rows*3, cols*3))

count = 0

conditions = ['Unc', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

for dev in list(datadic):
    for pil in list(datadic[dev]):

        for dis in list(datadic[dev][pil]):
            count += 1
            
            ax = plt.subplot(rows,cols,count)
     
            total_tracks = len(datadic[dev][pil][dis]['time'])
            print(total_tracks, 'tracks')
      
           
            for i in range(total_tracks):
                
                path = np.array([*datadic[dev][pil][dis]['rescaled_pos'][i]])                
                plt.plot(*path.T)
                ax.set_title(conditions[count-1], fontsize = 16)
                #plt.suptitle('Overlay of ' + f'{total_tracks}' + ' Tracks: Confinement Level '+ f'{pil}'[3] + '; Disorder Level '+ f'{dis}'[3], fontsize=12)
                plt.xlabel('µm',fontsize=14)
                plt.ylabel('µm',fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)

plt.tight_layout()  
plt.savefig('suppfig1_trackoverlays.png', dpi = 300)
plt.close()


# In[ ]:


#compute the # of tracks in all MSDs
N_tr = [] #create a list for all the tracks 
for dev in list(datadic): 
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):
            
            n_tracks = []
           
            for tr in range(len(datadic[dev][pil][dis]['MSD'])-1):
                n_tracks.append(len(datadic[dev][pil][dis]['MSD'][tr]))  # compute the MSD lengths for each track             
            #print(len(n_tracks))       
            N_tr.append(n_tracks) #A list of lists consisting the numbers of tracks for each condition

y_axhline = 50

lags = []
for n_tr in N_tr:

    new = [list(range(1,i+1)) for i in n_tr]
    flat_list = [item/fps for i in new for item in i]
    frequency_counts = Counter(flat_list)
    numbers, counts = zip(*sorted(frequency_counts.items()))

    for index, j in enumerate(list(counts)):            
        if j <= y_axhline:
            lag = numbers[index]
            lags.append(lag)
            break 
        if j > y_axhline and index == list(counts).index(list(counts)[-1]):
            lags.append(numbers[-1])
    #print(counts[-200:])

x_axvline = 20.0
cutoff = lags
#cutoffs based on the number of the tracks and the lagtime. 
for j, time in enumerate(cutoff):
    if time >= x_axvline:
        cutoff[j] = x_axvline
        
print(cutoff)

#Plot the individual MSDs and ensemble MSDs in subplots [Supplementary Figure 4]. 
mpl.rcParams['font.family'] = 'Arial'
rows = 4
cols = 4
plt.subplots_adjust(hspace=0.6)
plt.figure(figsize=(4*cols, 4*rows))
#plt.suptitle('Mean-squared Displacements', fontsize = 20)
count = 0

conditions = ['Unc', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C= 6 µm; D = 2', 'C= 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

meanofmeans_MSD = []
meanofstds_MSD = []

for dev in list(datadic):
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):
            
            count += 1
            ax = plt.subplot(rows,cols,count)
            
            mean_MSD = []
            std_MSD = []

            #print(len(datadic[dev][pil][dis]['time']))
            #print(len(datadic[dev][pil][dis]['MSD']))
            for tr in range(len(datadic[dev][pil][dis]['MSD'])-1):
                mean_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))
                std_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))
                #print(len(mean_MSD[-1]))
                for i in range(len(mean_MSD[-1])):
                    mean_MSD[-1][i] = np.mean(datadic[dev][pil][dis]['MSD'][tr][i])
                    std_MSD[-1][i] = np.std(datadic[dev][pil][dis]['MSD'][tr][i])
                    #print(len(datadic[dev][pil][dis]))
                    #print(mean_MSD[-1][:800])
                lagmax_global = cutoff[count-1]
                lagmax_local = len(mean_MSD[-1][1:])/fps
                #print(lagmax_local)
                time = min(lagmax_global,lagmax_local)
                #print(time)
                t = np.arange(0.05, time, 0.05)
                
                #print(len(mean_MSD[-1][1:int(lagmax_global*fps)]))
                plt.loglog(t, mean_MSD[-1][1:int(time*fps)]/mean_MSD[-1][1], color = 'lavender', label = 'Individual MSD', linewidth = 1,zorder=0)
                plt.xlabel('Lag time τ (s)',fontsize=22)
                plt.ylabel('MSD (µm²)',fontsize=22)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                #plt.loglog(std_MSD[-1], alpha = 1/len(datadic[dev][pil][dis]), color = 'blue')
            lagmax_global = cutoff[count-1]
            t = np.arange(0.05, lagmax_global, 0.05)
            meanofmeans_MSD.append(np.nanmean(np.array(list(it.zip_longest(*mean_MSD)),dtype=float),axis=1))
            emsd = np.array(meanofmeans_MSD[-1][1:int(lagmax_global*fps)])/meanofmeans_MSD[-1][1]
            #meanofstds_MSD.append(np.nanmean(np.array(list(it.zip_longest(*std_MSD)),dtype=float),axis=1))
            plt.loglog(t, emsd,color='dodgerblue',linewidth = 3, label = 'Mean of MSDs',zorder=1) 
            #plt.loglog(meanofstds_MSD[-1][:800],color='violet') 
            ax.set_title(conditions[count-1], fontsize=20)
            
            #Remove duplicate labels by putting them in a dictionary before calling legend()! :) Reference: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left')
            #plt.legend(loc='upper left')
plt.tight_layout() 
plt.savefig('suppfig4_indMSDeMSD.png', dpi = 300)
plt.close()


# In[ ]:


### Plot ensemble MSDs 6 um confinement [Figure 3a left]
mpl.rcParams['font.family'] = 'Arial'
colors = ['xkcd:grey'] + sns.color_palette('Blues_r') 
name = ['Unc', 'D = 0', 'D = 1',  'D = 2', 'D = 3']

fig,ax = plt.subplots(figsize=(8,6))
plt.title('6 µm',fontsize=18)
meanofmeans_MSD = []
meanofstds_MSD = []

count = 0
for dev in list(datadic):
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):
            
            mean_MSD = []
            std_MSD = []
            if (pil == 'pil0' and dis == 'dis0') or (pil == 'pil1' and dis == 'dis1') or (pil == 'pil1' and dis == 'dis2') or (pil == 'pil1' and dis == 'dis3') or (pil == 'pil1' and dis == 'dis4'):
                count += 1
                for tr in range(len(datadic[dev][pil][dis]['MSD'])-1):
                    mean_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))
                    std_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))

                    for i in range(len(mean_MSD[-1])):
                        mean_MSD[-1][i] = np.mean(datadic[dev][pil][dis]['MSD'][tr][i])
                        std_MSD[-1][i] = np.std(datadic[dev][pil][dis]['MSD'][tr][i])

                meanofmeans_MSD.append(np.nanmean(np.array(list(it.zip_longest(*mean_MSD)),dtype=float),axis=1))
                #meanofstds_MSD.append(np.nanmean(np.array(list(it.zip_longest(*std_MSD)),dtype=float),axis=1))

                lagmax_global = cutoff[count-1]
                t = np.arange(0.05, lagmax_global, 0.05)
                emsd = np.array(meanofmeans_MSD[-1][1:int(lagmax_global*fps)])/meanofmeans_MSD[-1][1]
                plt.loglog(t, emsd, label = name[count-1] , color=colors[count-1],linewidth=2,zorder=1) 
                plt.xlabel('Lag time τ (s)',fontsize=22)
                plt.ylabel('MSD (µm²)',fontsize=22)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))   
plt.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize=18)
plt.tight_layout()
plt.savefig('fig3a_eMSD_6um.png', transparent=True, dpi = 300) 
plt.close()

### Plot ensemble MSDs 2.6 um confinement [Figure 3c center]
plt.rcParams['font.family'] = 'Arial'
colors = ['xkcd:grey', 'xkcd:rust', 'xkcd:orange', 'xkcd:peach','xkcd:golden']

fig,ax = plt.subplots(figsize=(8,6))
plt.title('2.6 µm',fontsize=18)
meanofmeans_MSD = []
meanofstds_MSD = []

count = 0
for dev in list(datadic):
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):
            
            mean_MSD = []
            std_MSD = []
            if (pil == 'pil0' and dis == 'dis0') or (pil == 'pil2' and dis == 'dis1') or (pil == 'pil2' and dis == 'dis2') or (pil == 'pil2' and dis == 'dis3') or (pil == 'pil2' and dis == 'dis4'):
                count += 1
                for tr in range(len(datadic[dev][pil][dis]['MSD'])-1):
                    mean_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))
                    std_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))

                    for i in range(len(mean_MSD[-1])):
                        mean_MSD[-1][i] = np.mean(datadic[dev][pil][dis]['MSD'][tr][i])
                        std_MSD[-1][i] = np.std(datadic[dev][pil][dis]['MSD'][tr][i])
                        
                meanofmeans_MSD.append(np.nanmean(np.array(list(it.zip_longest(*mean_MSD)),dtype=float),axis=1))
                #meanofstds_MSD.append(np.nanmean(np.array(list(it.zip_longest(*std_MSD)),dtype=float),axis=1))

                lagmax_global = cutoff[count-1]
                t = np.arange(0.05, lagmax_global, 0.05)
                emsd = np.array(meanofmeans_MSD[-1][1:int(lagmax_global*fps)])/meanofmeans_MSD[-1][1]
                plt.loglog(t, emsd, label = name[count-1] , color=colors[count-1],linewidth=2,zorder=1) 
                plt.xlabel('Lag time τ (s)',fontsize=22)
                plt.ylabel('MSD (µm²)',fontsize=22)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))            
plt.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize=18)
plt.tight_layout()
plt.savefig('fig3a_eMSD_3um.png', transparent=True, dpi = 300) 
plt.close()

### Plot ensemble MSDs 1.3 um confinement [Figure 3a right]
plt.rcParams['font.family'] = 'Arial'
colors = ['xkcd:grey', 'xkcd:navy green', 'xkcd:dark cyan', 'xkcd:jade','xkcd:slime green']

fig,ax = plt.subplots(figsize=(8,6))
plt.title('1.3 µm',fontsize=18)
meanofmeans_MSD = []
meanofstds_MSD = []

count = 0
for dev in list(datadic):
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):
            
            mean_MSD = []
            std_MSD = []
            if (pil == 'pil0' and dis == 'dis0') or (pil == 'pil3' and dis == 'dis1') or (pil == 'pil3' and dis == 'dis2') or (pil == 'pil3' and dis == 'dis3') or (pil == 'pil3' and dis == 'dis4'):
                count += 1
                for tr in range(len(datadic[dev][pil][dis]['MSD'])-1):
                    mean_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))
                    std_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))

                    for i in range(len(mean_MSD[-1])):
                        mean_MSD[-1][i] = np.mean(datadic[dev][pil][dis]['MSD'][tr][i])
                        std_MSD[-1][i] = np.std(datadic[dev][pil][dis]['MSD'][tr][i])

                meanofmeans_MSD.append(np.nanmean(np.array(list(it.zip_longest(*mean_MSD)),dtype=float),axis=1))
                #meanofstds_MSD.append(np.nanmean(np.array(list(it.zip_longest(*std_MSD)),dtype=float),axis=1))

                lagmax_global = cutoff[count-1]
                t = np.arange(0.05, lagmax_global, 0.05)
                emsd = np.array(meanofmeans_MSD[-1][1:int(lagmax_global*fps)])/meanofmeans_MSD[-1][1]
                plt.loglog(t, emsd, label = name[count-1] , color=colors[count-1],linewidth=2,zorder=1) 
                plt.xlabel('Lag time τ (s)',fontsize=22)
                plt.ylabel('MSD (µm²)',fontsize=22)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))            
plt.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize=18)
plt.tight_layout()
plt.savefig('fig3a_eMSD_1um.png', transparent=True, dpi = 300)
plt.close()


# In[ ]:


##### Use power law to fit MSDs
# Define the power law function
def power_law(t, A, ν):
    return A * t**ν

rows = 4
cols = 4
plt.subplots_adjust(hspace=0.6)
plt.figure(figsize=(4*cols, 4*rows))

colors = ['xkcd:grey'] + sns.color_palette('Blues_r')[:4] + ['xkcd:dark tan', 'xkcd:orange', 'xkcd:peach','xkcd:golden'] +['xkcd:dark green', 'xkcd:grass', 'xkcd:jade','xkcd:slime green']

count = 0

cutofftimes = [3.5, 3.3, 2.9, 3.0, 2.4, 2.6, 2.3, 2.1, 1.8, 2.1, 1.5, 1.7, 1.3] 
M_msd = []
M_std = []
slopes_all = []

for dev in list(datadic):
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):
                
            count += 1

            ax = plt.subplot(rows,cols,count)

            mean_MSD = []

            for tr in range(len(datadic[dev][pil][dis]['MSD'])-1):
                mean_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))
                #std_MSD.append(np.zeros(len(datadic[dev][pil][dis]['MSD'][tr])))
                #print(len(mean_MSD[-1]))
                for i in range(len(mean_MSD[-1])):
                    mean_MSD[-1][i] = np.mean(datadic[dev][pil][dis]['MSD'][tr][i])
                    #std_MSD[-1][i] = np.std(datadic[dev][pil][dis]['MSD'][tr][i])
            #print(mean_MSD)                
            mean_MSD_transpose = list(map(list, it.zip_longest(*mean_MSD, fillvalue=np.nan)))

            #print(len(mean_MSD_transpose))
            time = cutofftimes[count-1]
            #lagmax_local = len(mean_MSD[-1][1:])/fps
            print(time)
            #time = min(lagmax_global,lagmax_local)
            #print(time)
            t = np.arange(0.05, time, 0.05)
            #print(len(t))

            msd = np.array(mean_MSD_transpose[1:int(time*fps)]) #transposed MSD list of lists (outer lists are time points and inner lists are trajectories)

            # Calculate mean and standard deviation
            msd_mean = np.nanmean(msd, axis=1)
            msd_std = np.nanstd(msd, axis=1)
            #print(len(msd_mean))
            M_msd.append(np.mean(msd_mean))
            M_std.append(np.mean(msd_std))
            plt.loglog(t, msd_mean/msd_mean[0], color=colors[count-1], linewidth=2,label='eMSD')

            # Plot error bands
            ax.fill_between(t, (msd_mean - msd_std)/msd_mean[0], (msd_mean + msd_std)/msd_mean[0], color=colors[count-1], alpha=0.2)
            ax.set_title(conditions[count-1], fontsize=20)
            #print(msd_mean)    
            plt.xlabel('Lag time τ (s)',fontsize=20)
            plt.ylabel('MSD (µm²)',fontsize=20)
            plt.ylim(1e-1,1e4)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            # Fit power law to initial part of the data
            fit_range = t < 0.5  # Consider lag times less than 1
            t_fit = t[fit_range]
            msd_mean_fit = msd_mean[:len(t_fit)]/msd_mean[0]

            popt, pcov = curve_fit(power_law, t_fit, msd_mean_fit, p0=[1, 2])
            A, ν = popt
            slopes_all.append(ν)
            plt.loglog(t_fit,power_law(t_fit,*popt),color='xkcd:magenta',linestyle='dashed', label=f'Fit ν={ν:.2f}')
            plt.legend(loc='upper left', fontsize=16)

plt.tight_layout()
plt.savefig('fig3b&suppfig5_MSDfit.png',transparent=True,dpi=300)
plt.close()


# In[ ]:


nv = []
for i in slopes_all:
    i = round(i,2)
    nv.append(i)

nv_tmp = [nv[0]]*3 + nv
nvs = nv_tmp[12:] + nv_tmp[8:12] + nv_tmp[4:8] + nv_tmp[:4]

# Updated confinement and disorder values
confinement = ['1.3 µm'] * 4 + ['2.6 µm'] * 4 + ['6 µm'] * 4 + ['Unc'] * 4
disorder = [0, 1, 2, 3] * 4

# Creating the DataFrame
df = pd.DataFrame({
    'C': confinement,
    'D': disorder,
    'ν': nvs
})

# Reorder the 'C' column to have rows from 'Unc' to '1.3 µm'
df['C'] = pd.Categorical(df['C'], categories=['Unc', '6 µm', '2.6 µm', '1.3 µm'], ordered=True)


# Pivot the DataFrame to structure it for a heatmap
htmp = df.pivot(index="C", columns="D", values="ν")

# Plotting the heatmap

mpl.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(5, 7))
sns.heatmap(htmp, vmin=1.68, vmax=1.90, annot=True,fmt='.2f',cmap="Blues", cbar=True,linewidth=0.8,cbar_kws={
                "orientation": "horizontal", "pad": 0.2,                   # Adjust the space between the heatmap and the colorbar
                "shrink": 0.7,                # Shrink the colorbar size
                "aspect": 50,                 # Adjust the aspect ratio of the colorbar
                "fraction": 0.1              # Fraction of the colorbar size relative to the plot
            },annot_kws={'size': 18})
plt.xlabel("D", fontsize=20)  # X-axis label size
plt.ylabel("C", fontsize=20)  # Y-axis label size

plt.xticks(fontsize=18)  # X-axis tick label size
plt.yticks(fontsize=18)

plt.savefig('fig3d_htmp.png', dpi=300,transparent=True)
plt.close()

