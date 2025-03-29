#!/usr/bin/env python
# coding: utf-8

##### The script is to classify bacterial motion by assigning the trajectory points to two states: swim and
##### stall, according to the manuscript; 
##### then to compute key metrics for Figure 2 & 3 and Supplementary Fig 2, 3, 6 & 7. Last changed March 28, 2025.

import os
import glob,csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import random
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from statsmodels.stats.multitest import multipletests
import itertools


# In[ ]:


#define a function to calculate the velocity
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


# In[ ]:


#define a function to calculate the angle
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


# In[ ]:


#define a function to classify run and tumble and compute the key metrics: run time, tumble time, run and tumble proportion
def macrostates(speed_thresh, angle_thresh, df: pd.DataFrame):
    ## The lists to record the positions
    runs = []
    tumbles = []
    uncluster = []

    ## The lists to record corresponding speeds and angles
    run_s = []
    run_a = []
    tumble_s = []
    tumble_a = []

    ## The lists to record the time stamps 
    run_t = []
    tumble_t = []

    ## The lists to record the run and tumble lengths
    run_ls = []
    run_dts = []
    tumble_dts = []

    run_start_point = None
    tumble_start_point = None

    for idx, row in df.iterrows():
        x = row['rescaled_pos'][0]
        y = row['rescaled_pos'][1]
        speed = row['speed']
        angle = row['angle']

        if pd.notna(speed) and pd.notna(angle) and speed >= speed_thresh and angle <= angle_thresh:
            runs.append(row['rescaled_pos'])
            run_t.append(row['rescaled_time'])
            run_s.append(row['speed'])
            run_a.append(row['angle'])

            if run_start_point is None:
                run_start_point = (x, y)
                run_start_time = row['rescaled_time']

            run_end_point = (x, y)
            run_end_time = row['rescaled_time']

            if tumble_start_point is not None:
                tumble_duration = tumble_end_time - tumble_start_time
                if tumble_duration != 0:
                    tumble_dts.append(tumble_duration)
                tumble_start_point = None  # Reset tumble_start_point after ending the tumble

        else:
            if pd.notna(speed) and pd.notna(angle) and speed < speed_thresh and angle >= 0:
                tumbles.append(row['rescaled_pos'])
                tumble_t.append(row['rescaled_time'])
                tumble_s.append(row['speed'])
                tumble_a.append(row['angle'])

                if tumble_start_point is None:
                    tumble_start_point = (x, y)
                    tumble_start_time = row['rescaled_time']

                tumble_end_point = (x, y)
                tumble_end_time = row['rescaled_time']

                if run_start_point is not None:
                    run_distance = np.sqrt((run_end_point[0] - run_start_point[0])**2 + (run_end_point[1] - run_start_point[1])**2)
                    run_duration = run_end_time - run_start_time
                    if run_distance != 0:
                        run_ls.append(run_distance)
                    if run_duration > 0:
                        run_dts.append(run_duration)
                    run_start_point = None  # Reset run_start_point after ending the run

            if pd.notna(speed) and pd.notna(angle) and speed > speed_thresh and angle > angle_thresh:
                uncluster.append(row['rescaled_pos'])

    # If a run was ongoing at the end of the loop, add the final segment
    if run_start_point is not None:
        run_distance = np.sqrt((run_end_point[0] - run_start_point[0])**2 + (run_end_point[1] - run_start_point[1])**2)
        run_duration = run_end_time - run_start_time
        run_ls.append(run_distance)
        if run_duration > 0:
            run_dts.append(run_duration)

    # If a tumble was ongoing at the end of the loop, add the final segment
    if tumble_start_point is not None:
        tumble_duration = tumble_end_time - tumble_start_time
        if tumble_duration > 0:
            tumble_dts.append(tumble_duration)

    return runs, tumbles, uncluster, run_t, tumble_t, run_s, tumble_s, run_a, tumble_a, run_dts, tumble_dts, run_ls


# In[ ]:


upperdir = '/'.join(os.getcwd().split('/')[:-1]) #define upper directory
lowerdir = '/sorted_tracks/' #define lower directory where all the data is located
files = glob.glob(upperdir + lowerdir + '*.csv') #grab all the files in their pathways 
files = sorted(files) #sort the files based on the naming
len(files)


#Include the velocity and angle calculations 
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
            time_start = time[0]
            datadic[device][pillar][disorder]['time'][dt].append(time)
            datadic[device][pillar][disorder]['rescaled_time'][dt].append([(time[i]-time_start)/fps for i in range(len(time))]) #rescale the time as well based on fps
            
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


###collect the data for all angles and speeds and store it in a DataFrame
v_theta = []
for dt in interval:
    for dev in list(datadic):
        for pil in list(datadic[dev]):
            for dis in list(datadic[dev][pil]):
                v_theta.extend([(dt, dev, pil, dis, pos, speed, angle, time) for pos, speed, angle, time 
                in zip(datadic[dev][pil][dis]['rescaled_pos'][dt], datadic[dev][pil][dis]['speed'][dt], 
                       datadic[dev][pil][dis]['angle'][dt], datadic[dev][pil][dis]['rescaled_time'][dt])])

df = pd.DataFrame(v_theta, columns=['dt', 'strain', 'confinement', 'disorder', 'p', 's', 'a','t'])

pillar_mapping = {'pil0': 'Unc', 'pil1': 'C = 6 µm', 'pil2': 'C = 2.6 µm', 'pil3': 'C = 1.3 µm'}
disorder_mapping = {'dis0': 'no pillar', 'dis1': 'D = 0', 'dis2': 'D = 1', 
                    'dis3': 'D = 2', 'dis4':'D = 3'}

# Replace the category names in the DataFrame
df['confinement'] = df['confinement'].replace(pillar_mapping)
df['disorder'] = df['disorder'].replace(disorder_mapping) 

#Merge the two columns to one 'condition' column
df['condition'] = df['confinement'] + '; ' + df['disorder']
#df.drop(['confinement', 'disorder'], axis=1, inplace=True)
sub_dfs = {dt: group for dt, group in df.groupby('dt')}

#explode the lists in the columns of speed, angle and rescaled time
df_exploded = sub_dfs[1].explode(list('psat'))
df_exploded['p'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded['s'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded['a'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded['t'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded = df_exploded.rename(columns={'p':'rescaled_pos', 's':'speed', 'a':'angle', 't':'rescaled_time'})
# df_exploded['condition'] = df_exploded['condition'].astype('category')


### compute mean speed and set up several speed and angle threshold candidates
average_velocity = df_exploded.groupby(['condition'])['speed'].mean().reset_index()
avg_vel = average_velocity.rename(columns={'speed':'Mean_Speed'})
mean_speed_unconfined = avg_vel.iloc[-1,-1]
half_mean_speed = round(mean_speed_unconfined/2,3)
angle_thresh = round(np.pi/3,3)
speed_choices = np.arange(half_mean_speed-3,half_mean_speed+4,1)
angle_choices = [round(np.pi/6,3),round(np.pi/5,3),round(5*np.pi/18,3),round(np.pi/3,3),round(2*np.pi/5,3),round(4*np.pi/9,3), round(np.pi/2,3)]

#Select the sub-dataset Unconfined

df_exploded['condition'] = df_exploded['condition'].astype('category')
unconfined = 'Unc; no pillar'
unconfined_df = df_exploded[(df_exploded['condition'] == unconfined)]

#Check how much data lies in the 'no-man land' (to be removed)


# for s in speed_choices: 
#     speed_angle=[]
#     for index, row in unconfined_df.iterrows():
#         speed = row['speed']
#         angle = row['angle']

#         if speed > s and angle > (np.pi/3):
#             speed_angle.append(False)
#         else: 
#             speed_angle.append(True)

#     count = 0
#     for i in speed_angle:

#         if i == False:      
#             count += 1


#     h_speed_h_angle = count/len(speed_angle)
#     print(format(h_speed_h_angle, ".2%"))


### speed-angle bivariate histogram for the unconfined region [Figure 2b]
mpl.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(figsize=(8.5,8))

ax.set_box_aspect(1)
speed_unconfined = []
angle_unconfined = []
for index, row in unconfined_df.iterrows():
    speed = row['speed']
    angle = row['angle']
    speed_unconfined.append(speed)
    angle_unconfined.append(angle)

kde = sns.kdeplot(x=angle_unconfined,y=speed_unconfined, cmap='Blues',fill=True, bw_adjust=0.8, cbar=True, cbar_kws={'shrink': 0.845, 'pad': 0.02, 'label': 'density'})

# Access the colorbar and format the tick labels
cbar = kde.figure.axes[-1]
cbar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

plt.xticks([0, np.pi/4, np.pi/2,3*np.pi/4, np.pi],[0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'],fontsize=14)
plt.xlabel('δθ (rad)',fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0,60)
plt.xlim(0,np.pi)
plt.plot(np.linspace(0,np.pi,20),np.array([half_mean_speed]*20),linestyle='dashed',linewidth=3,color='orange',label=
        'Speed threshold')
 
plt.plot(np.array([angle_thresh]*20),np.linspace(half_mean_speed,60,20),linestyle='dashed',linewidth=3,color='darkblue',label='Angle threshold')
plt.ylabel('Speed, v (µm/s)',fontsize=20)
plt.title('Unconfined',fontsize =20)
plt.legend(fontsize=20, loc='upper right')

plt.tight_layout() 
plt.savefig('fig2b_hist_bivariate.png',transparent=True, dpi=300)
#plt.close()

#### speed-angle jointplot for the unconfined region [partial Supplementary Fig 3]
# Set font
plt.rcParams['font.family'] = 'Arial'

speed_unconfined = []
angle_unconfined = []
for index, row in unconfined_df.iterrows():
    speed = row['speed']
    angle = row['angle']
    if speed > 0.0 and angle > 0.0:
        speed_unconfined.append(speed)
        angle_unconfined.append(angle)

# Create jointplot with histograms
g = sns.jointplot(
    x=angle_unconfined, y=speed_unconfined,
    kind="hist", cmap="plasma", 
    pthresh=.05, pmax=0.9, 
    marginal_kws={'color': 'xkcd:pale violet', 'edgecolor': 'black'},  # Marginal histogram styling
)

# Get the main Axes object for the jointplot
ax_main = g.ax_joint

# Add colorbar manually
cbar = plt.colorbar(ax_main.collections[0], ax=ax_main, pad=0.02, shrink=0.9)
cbar.set_label('Count', fontsize=12)
#cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

# Formatting the main plot
ax_main.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax_main.set_xticklabels([0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=14)
ax_main.tick_params(axis='y', labelsize=14)
ax_main.set_xlabel(r'$\delta\theta$ (rad)', fontsize=20)
ax_main.set_ylabel('Speed, v (µm/s)', fontsize=20)
ax_main.set_xlim(0, np.pi)
ax_main.set_ylim(0, 60)

# Add dashed threshold lines
ax_main.plot(np.linspace(0, np.pi, 20), np.array([half_mean_speed] * 20), linestyle='dashed', linewidth=3, color='gold', label='Speed threshold')
ax_main.plot(np.array([angle_thresh] * 20), np.linspace(0, 60, 20), linestyle='dashed', linewidth=3, color='xkcd:steel gray', label='Angle threshold')

# Add title and legend
g.fig.suptitle('Unconfined', fontsize=20,y=0.95)  # Move title above marginal plots
ax_main.legend(fontsize=14, loc='upper right')

# Show plot
plt.tight_layout()
#plt.savefig('jointplot_unconfined.png', transparent=True, dpi=300,bbox_inches='tight')
#plt.close()

### speed-angle jointplot for all investigated regions [Supplementary Fig 3] 
# Set font
mpl.rcParams['font.family'] = 'Arial'

# Loop through conditions and create plots
count = 0
for (condition, gp), ax in zip(df_exploded.groupby('condition'), axes):
    # Convert speed and angle to numeric
    gp['speed'] = pd.to_numeric(gp['speed'], errors='coerce')
    gp['angle'] = pd.to_numeric(gp['angle'], errors='coerce')

    # Filter valid values
    speed_valid = gp.loc[(gp['speed'] > 0) & (gp['angle'] > 0), 'speed']
    angle_valid = gp.loc[(gp['speed'] > 0) & (gp['angle'] > 0), 'angle']

    # Create jointplot with histograms
    g = sns.jointplot(
        x=angle_valid, y=speed_valid,
        kind="hist", cmap="plasma", 
        pthresh=.05, pmax=0.9,
        marginal_kws={'color': 'xkcd:pale violet', 'edgecolor': 'black'},  # Marginal histogram styling
    )

    # Get the main Axes object for the jointplot
    ax_main = g.ax_joint

    # Add colorbar manually
    cbar = plt.colorbar(ax_main.collections[0], ax=ax_main, pad=0.02, shrink=0.9)
    cbar.set_label('Count', fontsize=12)
    #cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

    # Formatting the main plot
    ax_main.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax_main.set_xticklabels([0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=14)
    ax_main.tick_params(axis='y', labelsize=14)
    ax_main.set_xlabel(r'$\delta\theta$ (rad)', fontsize=20)
    ax_main.set_ylabel('Speed, v (µm/s)', fontsize=20)
    ax_main.set_xlim(0, np.pi)
    ax_main.set_ylim(0, 60)

    # Add dashed threshold lines
    ax_main.plot(np.linspace(0, np.pi, 20), np.array([half_mean_speed] * 20), linestyle='dashed', linewidth=3, color='gold', label='Speed threshold')
    ax_main.plot(np.array([angle_thresh] * 20), np.linspace(0, 60, 20), linestyle='dashed', linewidth=3, color='xkcd:steel gray', label='Angle threshold')

    # Add title and legend
    g.fig.suptitle(f'{condition}', fontsize=20, y=0.95)  # Move title above marginal plots
    ax_main.legend(fontsize=14, loc='upper right')

    count += 1
#     if count >= rows * cols:  # Stop if we've plotted enough conditions
#         break

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the large plot
    plt.savefig(f'suppfig3_jointplot_{condition}.png', transparent=True, dpi=300)
    #plt.show()
    #plt.close()

### Sensitivity Check for Speed & Angle Thresholds
mpl.rcParams['font.family'] = 'Arial'
run_time = {}
run_lengths = {}
for speed in speed_choices:
    
    results = macrostates(speed,angle_thresh,unconfined_df)
    
    run_dts = results[9]
    
    run_ls = results[11]
    
    if speed in run_time:
        run_time[speed].extend([run_dts])
    else: 
        run_time[speed] = run_dts
    
    if speed in run_lengths:
        run_lengths[speed].extend([run_ls])
    else:
        run_lengths[speed] = run_ls

data1 = []
for speed,time in run_time.items():
    for t in time:
        data1.append({'speed':speed,'time':t})
df1 = pd.DataFrame(data1)

fig,ax=plt.subplots(figsize=(8,6))
sns.violinplot(data=df1,x='speed',y='time',palette='tab10')
plt.ylabel('run time (s)',fontsize=20)
plt.xlabel('speed (µm/s)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('suppfig2_t_vs_s.png',dpi=300)
#plt.show()
plt.close()

data2 = []
for speed,lengths in run_lengths.items():
    for l in lengths:
        data2.append({'speed':speed,'length':l})
df2 = pd.DataFrame(data2)

fig,ax=plt.subplots(figsize=(8,6))
sns.violinplot(data=df2,x='speed',y='length',palette='tab10')
plt.ylim(0,100)
plt.ylabel('run length (µm)',fontsize=20)
plt.xlabel('speed (µm/s)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('suppfig2_l_vs_s.png',dpi=300)
#plt.show()
plt.close()

run_time_1 = {}
run_lengths_1 = {}
for angle in angle_choices:
    
    results = macrostates(half_mean_speed,angle,unconfined_df)
    
    run_dts = results[9]
    
    run_ls = results[11]
    
    if angle in run_time_1:
        run_time_1[angle].extend([run_dts])
    else: 
        run_time_1[angle] = run_dts
    
    if speed in run_lengths_1:
        run_lengths_1[angle].extend([run_ls])
    else:
        run_lengths_1[angle] = run_ls

data3 = []
for angle,lengths in run_lengths_1.items():
    for l in lengths:
        data3.append({'angle':angle,'length':l})
df3 = pd.DataFrame(data3)

fig,ax=plt.subplots(figsize=(8,6))
sns.violinplot(data=df3,x='angle',y='length',palette='tab10')
plt.ylim(0,100)
plt.ylabel('run length (µm)',fontsize=20)
plt.xlabel('angle (rad)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('suppfig2_l_vs_a.png',dpi=300)
#plt.show()
plt.close()

data4 = []
for angle,time in run_time_1.items():
    for t in time:
        data4.append({'angle':angle,'time':t})
df4 = pd.DataFrame(data4)

fig,ax=plt.subplots(figsize=(8,6))
sns.violinplot(data=df4,x='angle',y='time',palette='tab10')
plt.ylabel('run time (s)',fontsize=20)
plt.xlabel('angle (rad)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('suppfig2_t_vs_a.png',dpi=300)
#plt.show()
plt.close()


# In[ ]:


#### Two selected sample trajectories from the unconfined condition [Figure 2c]
# Specified indices
mpl.rcParams['font.family'] = 'Arial'
random_indices = [11, 55]

# Group the DataFrame by its index
traj = unconfined_df.groupby(unconfined_df.index)

# Collect all the 'run' time data to find the global min and max
all_times = []

for index, sub_df in traj:
    if index in random_indices:
        run_or_tumble = macrostates(half_mean_speed, angle_thresh, sub_df)
        run = run_or_tumble[0]
        all_times.extend(np.arange(0, len(run)/fps, 1/fps))

# Determine the global minimum and maximum time
min_t = min(all_times)
max_t = max(all_times)

# Create a normalization object with the global min and max time
norm = mpl.colors.Normalize(vmin=min_t, vmax=max_t)

for index, sub_df in traj:

    if index == 11 or index == 55:
        
        run_or_tumble = macrostates(half_mean_speed, angle_thresh, sub_df)
        
        run = run_or_tumble[0]
        tumble = run_or_tumble[1]
        #uncluster = run_or_tumble[2]
        
        fig, ax = plt.subplots(figsize=(8,8))
   
        ax.set_box_aspect(1)
        path_run = np.array(run)
        path_tumble = np.array(tumble)
        #path_uncluster = np.array(uncluster)
        
        # Create a normalization object with the global min and max time
        norm = mpl.colors.Normalize(vmin=min_t, vmax=max_t)
        time =np.arange(0,len(path_run)/fps,1/fps)
        scatter = ax.scatter(path_run[:,0], path_run[:, 1], s=15, linewidth=1, c=time, norm=norm, cmap= 'Blues_r', zorder=1) #facecolors='none', edgecolors='xkcd:windows blue') # s is the size of the points
        plt.title(f'track id: {index}')
        plt.xlim(-20,420)
        plt.ylim(-20,420)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(scatter, cax=cax,shrink=0.75)
        cbar.set_label('Time (s)', fontsize=14)
        cbar.ax.tick_params(labelsize=20)
        scatter = ax.scatter(path_tumble[:, 0], path_tumble[:, 1],s=15, linewidth=1, color='xkcd:barney purple',zorder=2) # s is the size of the points
        
        plt.savefig(f'fig2c_unc_traj_{index}.png',dpi=300)
        #plt.show()
        plt.close()
        
random_indices = [11]
norm = mpl.colors.Normalize(vmin=0, vmax=657)
traj = unconfined_df.groupby(unconfined_df.index)
for index, sub_df in traj:
    if index in random_indices:
        
        run_or_tumble = macrostates(half_mean_speed, angle_thresh, sub_df)
        
        run = run_or_tumble[0]
        tumble = run_or_tumble[1]
        uncluster = run_or_tumble[2]
        
        fig, ax = plt.subplots(figsize=(8,8))
   
        ax.set_box_aspect(1)
        path_run = np.array(run)
        path_tumble = np.array(tumble)
        path_uncluster = np.array(uncluster)
        
        time =np.arange(0,len(path_run))
        scatter = ax.scatter(path_run[:, 0], path_run[:, 1], s=80, linewidth=2, c=time, norm=norm, cmap= 'Blues_r', zorder=1) #facecolors='none', edgecolors='xkcd:windows blue') # s is the size of the points
        plt.xlim(250,300)
        plt.ylim(240,290)
        plt.title(f'track id: {index}')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(scatter, cax=cax,shrink=0.75)
        cbar.set_label('Time (s)', fontsize=14)
        cbar.ax.tick_params(labelsize=20)
        scatter = ax.scatter(path_tumble[:, 0], path_tumble[:, 1],s=50, linewidth=2, color='xkcd:barney purple',zorder=2) # s is the size of the points
        #scatter = ax.scatter(path_uncluster[:, 0], path_uncluster[:, 1],s=20, facecolors='none', edgecolors='gray',zorder=0)
       
        plt.savefig(f'fig2c_traj{index}_enlarged.png',transparent = True, dpi=300)
        #plt.show()
        plt.close()


# In[ ]:


##### Sample trajectory (partial), state bar, and the speed-angle bivariate lineplot [Figure 2d]
### sample trajectory (partial)
mpl.rcParams['font.family'] = 'Arial'
for index, sub_df in traj:

    if index == 34:
        
        run_or_tumble = macrostates(half_mean_speed, angle_thresh, sub_df)
        
        run = run_or_tumble[0]
        tumble = run_or_tumble[1]
        uncluster = run_or_tumble[2]
        run_t = run_or_tumble[3]
        tumble_t = run_or_tumble[4]
        
        fig, ax = plt.subplots(figsize=(8,8))
   
        ax.set_box_aspect(1)
        path_run = np.array(run)
        path_tumble = np.array(tumble)
        path_uncluster = np.array(uncluster)
        
        # Create a normalization object with the global min and max time
        norm = mpl.colors.Normalize(vmin=min_t, vmax=max_t)
        time =np.arange(0,len(path_run)/fps,1/fps)
        scatter = ax.scatter(path_run[:364, 0], path_run[:364, 1], s=15, linewidth=1, c=time[:364], norm=norm, cmap= 'Blues_r', zorder=1) #facecolors='none', edgecolors='xkcd:windows blue') # s is the size of the points
        plt.title(f'track id: {index} (partial)')
        plt.xlim(-20,420)
        plt.ylim(-20,420)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(scatter, cax=cax,shrink=0.75)
        cbar.set_label('Time (s)', fontsize=14)
        cbar.ax.tick_params(labelsize=20)
        scatter = ax.scatter(path_tumble[:32, 0], path_tumble[:32, 1],s=15, linewidth=1, color='xkcd:barney purple',zorder=2) # s is the size of the points
        #scatter = ax.scatter(path_uncluster[:, 0], path_uncluster[:, 1],s=15, color='grey',zorder=3)
        
        plt.savefig(f'fig2d_traj.png',transparent=True, dpi=300)
        #plt.show()
        plt.close()
        
###plot the state bar
# Specified indices
random_indices = [34]

# Group the DataFrame by its index
traj = unconfined_df.groupby(unconfined_df.index)

# Collect the sub-dataframes corresponding to the specified indices
selected_dfs = []

for index, sub_df in traj:
    if index in random_indices:
        selected_dfs.append(sub_df)

# Concatenate the selected sub-dataframes into a new DataFrame
traj34 = pd.concat(selected_dfs)

run_or_tumble = macrostates(half_mean_speed, angle_thresh, traj34) #runs, tumbles, uncluster, run_t, tumble_t, run_s, tumble_s, run_a, tumble_a, run_ls

run = run_or_tumble[0]
tumble = run_or_tumble[1]
uncluster = run_or_tumble[2]

run_t = run_or_tumble[3]
tumble_t = run_or_tumble[4]

run_speed = run_or_tumble[5]
tumble_speed = run_or_tumble[6]

run_angle = run_or_tumble[7]
tumble_angle = run_or_tumble[8]

run_dts = run_or_tumble[9]
tumble_dts = run_or_tumble[10]
run_ls = run_or_tumble[11]

time_combined = np.concatenate((run_t[:364],tumble_t[:32]))
path_combined = np.concatenate((run[:364], tumble[:32]))

# Create a custom color array
colors = []
for t in time_combined:
    if t in run_t:
        norm_time = (t - min(run_t)) / (max(run_t) - min(run_t))
        colors.append(plt.cm.Blues_r(norm_time))
    else:
        colors.append('xkcd:barney purple')

norm = mpl.colors.Normalize(vmin=min(run_t), vmax=max(run_t))

# Create the horizontal bar plot to represent time intervals
fig, ax = plt.subplots(figsize=(10, 0.6))

plt.scatter(time_combined,[2]*len(time_combined),color=colors, marker='|',linewidth=3, s=200)

ax.set_xlabel('Time (s)')
plt.xticks(np.arange(0.0,22.0,2.0),fontsize=20) 
ax.get_yaxis().set_visible(False)
plt.subplots_adjust(bottom=0.5)
plt.savefig('fig2d_bar.png',bbox_inches='tight', transparent = True, dpi=300)
#plt.show()
plt.close()

### speed-angle bivariate 
#defined a function to extract the speed, angle, timepoint info
def speedangle(speed_thresh, angle_thresh, df: pd.DataFrame):
    
    ## The lists to record the positions
    s = []
    a = []
    time = []
    unc_s = []
    unc_a = []
    
    start_point = None

    for idx, row in df.iterrows():

        speed = row['speed']
        angle = row['angle']
        t = row['rescaled_time']
      
        if pd.notna(speed) and pd.notna(angle) and speed > speed_thresh and angle > angle_thresh:
            
            unc_s.append(speed)
            unc_a.append(angle)
            
        else:
            if pd.notna(speed) and pd.notna(angle) and speed > 0 and angle > 0:
                s.append(speed)
                a.append(angle)
                time.append(t)
    
    return s,a,time

s,a,time = speedangle(half_mean_speed, angle_thresh, traj34)
tmax = traj34['rescaled_time'].iloc[-1]
fig, ax1 = plt.subplots(figsize=(10,5))

#Speed Portion
color1 = 'xkcd:black'
ax1.plot(time[:396],s[:396],color=color1, zorder = 4,label='Speed') 

ax1.set_xlabel('Time (s)',fontsize=20)
ax1.set_ylabel('Speed (um/s)',color=color1, fontsize=20)
ax1.set_xlim(0,20.0)
plt.xticks(fontsize=20)
ax1.set_yticks(np.arange(0,120,20),[0,20,40,60,80,''])
ax1.set_ylim(0,100)
plt.yticks(fontsize=20)
ax1.tick_params(axis='y', labelcolor=color1)
plt.legend(loc='upper left',fontsize='large')

ax2 = ax1.twinx()

color2 = 'xkcd:dark grey'
ax2.plot(time[:396],a[:396],color=color2, linestyle='solid',zorder=0, alpha=0.4, label='Turn angle')
#ax2.scatter(unc_t,unc_a,color='red',s=10, zorder=2)

ax2.set_ylabel('δθ (rad)',color=color2, fontsize=20)
ax2.set_yticks(np.arange(0,6/4*np.pi,1/4*np.pi),[0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$',r'$5\pi/4$'])
plt.yticks(fontsize=20)
ax2.tick_params(axis='y', labelcolor=color2)
plt.legend(loc='upper center',fontsize='large')
plt.savefig('fig2d_bivariate_lineplot.png',bbox_inches='tight', transparent = True, dpi=300)
#plt.show()
plt.close()


# In[ ]:


##### Compute and store the key metrics: run time (swim time), tumble time (stall time), run (swim) proportion & tumble (stall) proportion
def convert_to_numeric(pos):
    if isinstance(pos, list):
        return [pd.to_numeric(coord, errors='coerce') for coord in pos]
    return pos

def flatten(matrix):
    return [item for row in matrix for item in row]

df_exploded['particle_id'] = df_exploded.index
gp_dfs = df_exploded.groupby(['condition', 'particle_id'])

# Create an empty dictionary to store aggregated run lengths for each condition
run_t_all = {}
tumble_t_all = {}

run_dts_all = {}
tumble_dts_all = {}

for (condition, particle_id), gp in gp_dfs:
    
    gp['rescaled_pos'] = gp['rescaled_pos'].apply(convert_to_numeric)
    gp['speed'] = gp['speed'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    gp['angle'] = gp['angle'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    gp['rescaled_time'] = gp['rescaled_time'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    run_or_tumble = macrostates(half_mean_speed, angle_thresh, gp)
    
    run_t = run_or_tumble[3]
    tumble_t = run_or_tumble[4]
    
    run_dts = run_or_tumble[9]
    tumble_dts = run_or_tumble[10]
    
    
    
    if condition in run_t_all:
        run_t_all[condition].extend([run_t])
    else:
        run_t_all[condition] = [run_t]
    
    if condition in tumble_t_all:
        tumble_t_all[condition].extend([tumble_t])
    else:
        tumble_t_all[condition] = [tumble_t]
        
    if condition in run_dts_all:
        run_dts_all[condition].extend([run_dts])
    else:
        run_dts_all[condition] = [run_dts]
        
    if condition in tumble_dts_all:
        tumble_dts_all[condition].extend([tumble_dts])
    else:
        tumble_dts_all[condition] = [tumble_dts]


### the dataset for mean run time
data5 = []

for condition, run_dts in run_dts_all.items():
    
    durations = flatten(run_dts)
    
    for duration in durations:
        
        data5.append({'Condition': condition, 'Run_T': duration})
        
df_run_dts = pd.DataFrame(data5)

### the dataset for mean tumble time
data6 = []

for condition, tumble_dts in tumble_dts_all.items():
    
    durations = flatten(tumble_dts)
    
    for duration in durations:
        
        data6.append({'Condition': condition, 'Tumble_T': duration})
        
df_tumble_dts = pd.DataFrame(data6)

#### Boxplot: Swim time across conditions [Figure 3a]
# Set font
plt.rcParams['font.family'] = 'Arial'

# Define the desired order
desired_order = [
    'Unc; no pillar',
    'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3',
    'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3',
    'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']
 
plt.figure(figsize=(8, 6))
sns.boxplot(x='Condition', y='Run_T', data=df_run_dts, order=desired_order, palette='Blues',width=0.6,
            showfliers=False, showmeans=True,meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":10})

plt.ylim(0,3.2)
plt.ylabel('Swim time (s)',fontsize=24)
plt.xlabel('Condition',fontsize=18)
plt.xticks(rotation=45, fontsize=16)  # Rotate x-axis labels if necessary
plt.yticks(fontsize=22)
plt.title('Swim time across conditions',fontsize=24)
plt.savefig('fig3_swim_time.pdf', transparent=True, dpi=300)
plt.show()

#### Boxplot: Stall time across conditions [Figure 3b]
# Set font
plt.rcParams['font.family'] = 'Arial'

# Define the desired order
desired_order = [
    'Unc; no pillar',
    'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3',
    'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3',
    'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']
 
plt.figure(figsize=(8, 6))
sns.boxplot(x='Condition', y='Tumble_T', data=df_tumble_dts, order=desired_order, palette='Reds',
            showfliers=False, width=0.6, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":10})

plt.ylim(0,1.3)
plt.ylabel('Stall time (s)',fontsize=24)
plt.xlabel('Condition',fontsize=18)
plt.xticks(rotation=45, fontsize=16)  # Rotate x-axis labels if necessary
plt.yticks(fontsize=22)
plt.title('Stall time across conditions',fontsize=24)
plt.savefig('fig3b_stall_time.pdf', transparent=True, dpi=300)
plt.show()

#### Significance test of swim times and plot a heatmap for the pair-wise comparisons [Supplementary Fig 6] 
# Get all unique condition pairs
conditions = df_run_dts['Condition'].unique()
pairs = list(itertools.combinations(conditions, 2))  # Generate all pairwise combinations

# Store p-values for significance testing
p_values_run = {}

for cond1, cond2 in pairs:
    # Extract run times for each condition
    runs_1 = df_run_dts[df_run_dts['Condition'] == cond1]['Run_T']
    runs_2 = df_run_dts[df_run_dts['Condition'] == cond2]['Run_T']
    
    # Check normality using Shapiro-Wilk test
    _, p1 = stats.shapiro(runs_1)
    _, p2 = stats.shapiro(runs_2)
    
    if p1 > 0.05 and p2 > 0.05:  # If both are normally distributed
        # Use independent t-test
        print(True)
        stat, p_val = stats.ttest_ind(runs_1, runs_2, equal_var=False)  # Welch’s t-test
    else:
        # Use Mann-Whitney U test (non-parametric)
        stat, p_val = stats.mannwhitneyu(runs_1, runs_2, alternative='two-sided')
    
    p_values_run[(cond1, cond2)] = p_val

# Display results
# for (cond1, cond2), p_val in p_values_run.items():
#     significance = 'Significant' if p_val < 0.05 else 'Not Significant'
#     print(f"Comparison {cond1} vs {cond2}: p-value = {p_val:.5f} ({significance})")
    
# Apply Bonferroni correction
p_adjusted = multipletests(list(p_values_run.values()), method='bonferroni')[1]

# Display corrected p-values
# for (cond1, cond2), adj_p in zip(p_values_run.keys(), p_adjusted):
#     significance = 'Significant' if adj_p < 0.05 else 'Not Significant'
#     print(f"Bonferroni-corrected {cond1} vs {cond2}: p-value = {adj_p:.4f} ({significance})")
    
mpl.rcParams['font.family'] = 'Arial'
# Get all unique conditions
conditions = df_run_dts['Condition'].unique()
pairs = list(itertools.combinations(conditions, 2))

# Initialize p-values dictionary
p_values_run = {}

# Compute p-values for each pair
for cond1, cond2 in pairs:
    runs_1 = df_run_dts[df_run_dts['Condition'] == cond1]['Run_T']
    runs_2 = df_run_dts[df_run_dts['Condition'] == cond2]['Run_T']
    
    # Check normality
    _, p1 = stats.shapiro(runs_1)
    _, p2 = stats.shapiro(runs_2)
    
    if p1 > 0.05 and p2 > 0.05:  
        stat, p_val = stats.ttest_ind(runs_1, runs_2, equal_var=False)  # Welch’s t-test
    else:
        stat, p_val = stats.mannwhitneyu(runs_1, runs_2, alternative='two-sided')
    
    p_values_run[(cond1, cond2)] = p_val

# Apply Bonferroni correction
p_adjusted = multipletests(list(p_values_run.values()), method='bonferroni')[1]
adjusted_p_values = dict(zip(p_values_run.keys(), p_adjusted))

# Convert to DataFrame for heatmap
p_matrix = pd.DataFrame(np.nan, index=conditions, columns=conditions)

for (cond1, cond2), p_val in adjusted_p_values.items():
    p_matrix.loc[cond1, cond2] = round(p_val,4)
    p_matrix.loc[cond2, cond1] = round(p_val,4)  # Fill symmetric values

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(p_matrix, annot=True, cmap="coolwarm_r", vmax=0.06, linewidths=0.5, cbar_kws={'label': 'Bonferroni-adjusted p-value', 'ticks': [0, 0.01, 0.02, 0.03, 0.04, 0.05]})
#plt.title("Heatmap of pairwise Mann-Whitney U test for swim times")
plt.xticks(rotation=45, ha='right',fontsize=12) # Rotate the x axis labels
plt.yticks(rotation=0,fontsize=12)
plt.tight_layout()
plt.savefig('suppfig6_sigtest_swimtime.png',dpi=500,transparent=True)
plt.show() 

#### Significance test of stall times and plot a heatmap for the pair-wise comparisons [Supplementary Fig 7]
mpl.rcParams['font.family'] = 'Arial'
# Get all unique condition pairs
conditions = df_tumble_dts['Condition'].unique()
pairs = list(itertools.combinations(conditions, 2))  # Generate all pairwise combinations

# Store p-values for significance testing
p_values_tumble = {}

for cond1, cond2 in pairs:
    # Extract run times for each condition
    tumble_1 = df_tumble_dts[df_tumble_dts['Condition'] == cond1]['Tumble_T']
    tumble_2 = df_tumble_dts[df_tumble_dts['Condition'] == cond2]['Tumble_T']
    
    # Check normality using Shapiro-Wilk test
    _, p1 = stats.shapiro(tumble_1)
    _, p2 = stats.shapiro(tumble_2)
    
    if p1 > 0.05 and p2 > 0.05:  # If both are normally distributed
        # Use independent t-test
        print(True)
        stat, p_val = stats.ttest_ind(tumble_1, tumble_2, equal_var=False)  # Welch’s t-test
    else:
        
        # Use   (non-parametric)
        stat, p_val = stats.mannwhitneyu(tumble_1, tumble_2, alternative='two-sided')
    
    p_values_tumble[(cond1, cond2)] = p_val

# Display results
# for (cond1, cond2), p_val in p_values_tumble.items():
#     significance = 'Significant' if p_val < 0.05 else 'Not Significant'
#     print(f"Comparison {cond1} vs {cond2}: p-value = {p_val:.5f} ({significance})")
    
# Apply Bonferroni correction
p_adjusted = multipletests(list(p_values_tumble.values()), method='bonferroni')[1]
adjusted_p_values = dict(zip(p_values_run.keys(), p_adjusted))

# Convert to DataFrame for heatmap
p_matrix = pd.DataFrame(np.nan, index=conditions, columns=conditions)

for (cond1, cond2), p_val in adjusted_p_values.items():
    p_matrix.loc[cond1, cond2] = round(p_val,4)
    p_matrix.loc[cond2, cond1] = round(p_val,4)  # Fill symmetric values

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(p_matrix, annot=True, cmap="coolwarm_r", vmax=0.06, linewidths=0.5, cbar_kws={'label': 'Bonferroni-adjusted p-value', 'ticks': [0, 0.01, 0.02, 0.03, 0.04, 0.05]})
#plt.title("Heatmap of pairwise Mann-Whitney U test for stall times")
plt.xticks(rotation=45, ha='right',fontsize=12)
plt.yticks(rotation=0,fontsize=12)
plt.tight_layout()
plt.savefig('suppfig7-sigtest-stalltime.png',dpi=500,transparent=True)
plt.show()

#### compute mean tumble (stall) bias [Figure 3c]
Tumbles = []
for condition, t in tumble_t_all.items():
    tumbles = []
    for bac in t:
        tumbles.append(len(bac))
    Tumbles.append(tumbles)

Runs = []
for condition, t in run_t_all.items():
    runs = []
    for bac in t:
        runs.append(len(bac))
    Runs.append(runs)
    
TB_all = []
for i,j in zip(Tumbles,Runs):
    TB = []
    for t,r in zip(i,j):
        tb = t / (t+r)
        TB.append(tb)
    TB_all.append(np.mean(TB))
    
TBs_all = TB_all + [np.nan] * 3

TBs = []
for i in TBs_all:
    i = round(i,3)
    TBs.append(i)

# Assign the tumble bias values to confinement and disorder
confinement = ['1.3 µm'] * 4 + ['2.6 µm'] * 4 + ['6 µm'] * 4 + ['Unc'] * 4
disorder = [0, 1, 2, 3] * 4

# Creating the DataFrame
df = pd.DataFrame({
    'C': confinement,
    'D': disorder,
    'TB': TBs
})

# Reorder the 'C' column to have rows from 'Unc' to '1.3 µm'
df['C'] = pd.Categorical(df['C'], categories=['Unc', '6 µm', '2.6 µm', '1.3 µm'], ordered=True)

# Pivot the DataFrame to structure it for a heatmap
htmp = df.pivot(index="C", columns="D", values="TB")

# Plotting the heatmap

mpl.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(5, 7))
sns.heatmap(htmp, annot=True, vmin=0.18, vmax=0.6, cmap="plasma", cbar=True,linewidth=0.8,cbar_kws={
                "orientation": "horizontal",  # Place the colorbar on top
                "pad": 0.2,                   # Adjust the space between the heatmap and the colorbar
                "shrink": 0.7,                # Shrink the colorbar size
                "aspect": 50,                 # Adjust the aspect ratio of the colorbar
                "fraction": 0.1              # Fraction of the colorbar size relative to the plot
            },annot_kws={'size': 18})
plt.xlabel("D", fontsize=18)  # X-axis label size
plt.ylabel("C", fontsize=18)  # Y-axis label size
plt.title('Tumble bias')
plt.xticks(fontsize=18)  # X-axis tick label size
plt.yticks(fontsize=18)
plt.savefig('fig3c_htmp_tb.png', dpi=300,transparent=True)
plt.close()
