#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### The script is written to compute velocity autocorrelation function and crossover time
##### for Supplementary Fig 6 (6a & 6b).

import os
import glob,csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy


# In[5]:


### Define the function for velocity calculation ###

def calculate_velocity(x_positions, y_positions, time_positions,interval:int,fps):
    # Calculate the differences in positions and time
    dx = np.array(x_positions[interval:]) - np.array(x_positions[:-interval])
    dy = np.array(y_positions[interval:]) - np.array(y_positions[:-interval])
    dt = (np.array(time_positions[interval:]) - np.array(time_positions[:-interval]))/fps
    
    # Calculate velocity components vx and vy
    vx = dx / dt
    vy = dy / dt
    
    # Return velocity vectors as a list of arrays
    velocity_vectors = [np.array([vx[i], vy[i]]) for i in range(len(vx))]
    
    return velocity_vectors


# In[6]:


### Load the dataset and create a dictionary to store the data ###

upperdir = '/'.join(os.getcwd().split('/')[:-3]) #define upper directory
lowerdir = '/TRACKS!/Sorted_Tracks/' #define lower directory where all the data is located
files = glob.glob(upperdir + lowerdir + '*.csv') #grab all the files in their pathways 
files = sorted(files) #sort the files based on the naming
len(files)


# In[12]:


datadic = {} #set up a data directory
dev = [] #number of device
pil= [] #number of confinement
dis = [] #number of disorders
rep = [] #number of repititions 
pixel_to_micron = 0.656 #pixel to micron
fps = 20 #frames per second
interval = 1
step = 160

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
        datadic[device][pillar][disorder]['vel_dot']=[[]]
        
    for i in range(len(track_names)):
        time = np.array(df[df['ID']==track_names[i]]['POSITION_T'].tolist())

        datadic[device][pillar][disorder]['time'].append(time)
        datadic[device][pillar][disorder]['rescaled_time'].append([time[i]/fps for i in range(len(time))]) #rescale the time as well based on fps
        x_pos = np.array(df[df['ID']==track_names[i]]['POSITION_X'].tolist())
        y_pos = np.array(df[df['ID']==track_names[i]]['POSITION_Y'].tolist())
        
        x_start = x_pos[0]
        y_start = y_pos[0]
        
        x_pos_rescaled = (x_pos-x_start)*pixel_to_micron
        y_pos_rescaled = (y_pos-y_start)*pixel_to_micron
      
        datadic[device][pillar][disorder]['pos'].append([(x_pos[t]-x_start,y_pos[t]-y_start) for t in range(len(x_pos))])
        datadic[device][pillar][disorder]['rescaled_pos'].append([(pixel_to_micron*x_pos[t], pixel_to_micron*y_pos[t]) for t in range(len(x_pos))]) 
        
        #Calculate the dot products of velocities 
        velocities = calculate_velocity(x_pos_rescaled, y_pos_rescaled, time, interval, fps) 
#         print(x_pos[:5],'x position')
#         print(y_pos[:5],'y position')
#         print(x_pos_rescaled[:5], 'x rescaled')
#         print(y_pos_rescaled[:5], 'y rescaled')
        
#         print(velocities[:5], 'velocity')
        #avg_v = avg_speed(velocities)
        
        #print(len(velocities),'len vel')
        
        if len(time) < 160:
            continue
        if len(time) > len(datadic[device][pillar][disorder]['vel_dot'][-1]):
            add_on = len(time) - len(datadic[device][pillar][disorder]['vel_dot'][-1])
            for j in range(add_on-1):
                datadic[device][pillar][disorder]['vel_dot'][-1].extend([[]])
        
        ranges = [(x, min(x+step,len(velocities))) for x in range(0, len(velocities), step)]
        
        for start, end in ranges:
            
            for shift in range(start, end):
                #if shift == 1:
                       #print((r[time+shift] - r[time])**2)
                    
                dot = np.dot(velocities[start], velocities[shift])
                
                #dot = np.dot((velocities[t0] - avg_v),(velocities[t0+shift] - avg_v)) #vx, vy
                
                datadic[device][pillar][disorder]['vel_dot'][-1][shift].extend([dot])       
        
        #print(t0, 'time')
#         for par in datadic[device][pillar][disorder]['vel_dot'][-1]:
#             print(par)
#             for i, lag in enumerate(par):
#                 print(lag)
                
        datadic[device][pillar][disorder]['vel_dot'].extend([[]])

    # Test whether the trajecories are stored properly           
#     if device == 'dev1' and pillar == 'pil0' and disorder == 'dis0':
#          for i in range(len(datadic[device][pillar][disorder]['rescaled_pos'])):
#             path = np.array(datadic[device][pillar][disorder]['rescaled_pos'][i])
#             plt.plot(*path.T) # ".T" attribute is used to transpose the array, swapping its rows and columns. # "*" unpacks the rows of the transposed array
#             plt.xlabel('µm')
#             plt.ylabel('µm')
    
#     plt.show()


# In[28]:


### Define a sliding window for velocity autocorrelation function ###

def sliding_window(elements, window_size, overlap):
    means = []
    step = window_size - overlap
    if len(elements) <= window_size:
        return elements
    for i in range(0, len(elements) - window_size +1,step):
        mean = np.mean(elements[i:i+window_size])
        means.append(mean)
        #print((elements[i:i+window_size]),'k')
    return means

# Example usage
a =[1,0.8,0.6,0.4,0.2,0.1,0.1]
window_size = 5
overlap = 3

result = sliding_window(a, window_size, overlap)
#print(result)


# In[94]:


### Plot velocity autocorrelation function ###

rows = 4
cols = 4
#plt.subplots_adjust(hspace=0.6)
plt.figure(figsize=(rows*3, cols*3))
count1 = 0

colors = ['xkcd:grey'] + sns.color_palette('Blues_r')[:4] + ['xkcd:dark tan', 'xkcd:orange', 'xkcd:peach','xkcd:golden'] +['xkcd:dark green', 'xkcd:grass', 'xkcd:jade','xkcd:slime green']

mean_time_corr = []

name = ['Unc', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C= 6 µm; D = 2', 'C= 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

window_size = 15
overlap = 5

crossover = []

for dev in list(datadic):
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):
            #print(len(datadic[dev][pil][dis]['time']))
            
            count = 0
            count1 += 1
        
            ax = plt.subplot(rows,cols,count1)
            
            max_t = max(len(tr) for tr in datadic[dev][pil][dis]['time'])
            
            #print(max_t)
            
            # Pad the vel_dot list to have the same lengths in the sublist (Note: list[sublist]])
            
            max_t0 = max(len(sublist) for sublist in datadic[dev][pil][dis]['vel_dot'])
            
            #print(max_t0)
            
#             for sublist in datadic[dev][pil][dis]['vel_dot']:
#                 #print(len(sublist))
            l = []
            for sublist in datadic[dev][pil][dis]['vel_dot']:
                combined_list = []
                
                for innerlist in sublist:

                    combined_list.extend(innerlist)
                    
                l.extend([combined_list])
           
            padded_vel_dot = [sublist+ [np.nan]*(max_t0-len(sublist))
                              for sublist in l]
            
            padded_vel_dot = np.array(padded_vel_dot)
            
            ranges = [(x, x+step) for x in range(0, max_t, step)]       
            #print(ranges)
            
            time_corrs = []
            for start, end in ranges:
                count += 1
                
                time_corr = []
                for time in range(start,end-1):
                            
                #print(time,'time')
                    top = padded_vel_dot[:-1,time]
                    #print(len(top))
                    #print(top,'toooooooooop')
                    bot = padded_vel_dot[:-1,start]

                    topsum = np.nansum(top, axis=0)
                    #print(topsum, 'topsum')

                    botsum = np.nansum(bot, axis=0)
                    #print(botsum, 'botsum')
                    
                    time_corr.append(topsum / botsum)
                    
                t = np.arange(0.05, step/fps, 0.05)
                
                plt.plot(t, time_corr, color='k',alpha=0.1,zorder=0)
                
                #print(time_corr,'result')
                time_corrs.append(time_corr)

            mean_cr = np.nanmean(time_corrs,axis=0)
            #print(mean_cr)
            mean = sliding_window(mean_cr,window_size,overlap)
            #print(len(mean))
            for idx, value in enumerate(mean_cr):
                if value <= 0.1:
                    crossover.append(idx/fps)
                    #crossover.append((window_size+idx*(window_size-overlap))/fps)
                    break
            
            plt.plot(t, np.nanmean(time_corrs,axis=0), color=colors[count1-1], label='averaged',linewidth=2,zorder=2)
            
            plt.plot(t,[0.0]*len(t),color='grey',linestyle='dashed',zorder=1)
            plt.ylim(-0.2,1.0)
            plt.xlim(-0.1,8.1)
            plt.xticks(np.arange(0.0, 8.1, step=2.0)) 
            plt.xlabel('Lag time τ (s)',fontsize=15)
            plt.ylabel('Cr (τ)',fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            plt.legend(fontsize='large')
            ax.set_title(f'{name[count1-1]}',fontsize=15)

plt.tight_layout() 
plt.savefig('suppfig6a-VACF.pdf',transparent=True,dpi=300)


# In[ ]:


### Plot crossover time v.s. condition ### 

f = plt.figure(figsize=(4,4),dpi=300)
ax = f.add_subplot(111)
#ax.yaxis.tick_right()
plt.scatter(crossover[::-1],name[::-1],color=colors[::-1])
plt.title('Crorrelation time',fontsize=12)
plt.xlabel('$T_{corr}$ (s)')
plt.xlim(0.5,3.5)
plt.savefig('suppfig6b-T_corr.pdf',transparent=True,dpi=300,bbox_inches='tight')

