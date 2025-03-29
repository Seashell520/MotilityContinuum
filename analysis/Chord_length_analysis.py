#!/usr/bin/env python
# coding: utf-8

##### The script is written by Leone V. Luzzatto and Haibei Zhang to compute chord lengths and swim (run) lengths and compare conditioned chord lengths with swim (run) lengths. It outputs Figure 7. Last changed March 28, 2025.

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
import glob
import itertools as itt
import skimage as skm


# ### Define a function to calculate velocities

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


# ### Define a function to calculate re-orientation angles

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


# ### Define a function to calculate run lengths

#Assume the dataframe has the rescaled x,y position data, correpsonding speed, angle and time data:

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
      
        if pd.notna(speed) and  pd.notna(angle) and speed > speed_thresh and angle < angle_thresh:
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


# ### Load Data & Apply Run Length Function

# Retrieve the data
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


#collect the data for all angles and speeds and store it in a DataFrame
v_theta = []
for dt in interval:
    for dev in list(datadic):
        for pil in list(datadic[dev]):
            for dis in list(datadic[dev][pil]):
                v_theta.extend([(dt, dev, pil, dis, pos, speed, angle) for pos, speed, angle 
                                in zip(datadic[dev][pil][dis]['rescaled_pos'][dt], 
                                       datadic[dev][pil][dis]['speed'][dt], datadic[dev][pil][dis]['angle'][dt])])

df = pd.DataFrame(v_theta, columns=['dt', 'strain', 'confinement', 'disorder', 'p', 's', 'a'])

pillar_mapping = {'pil0': 'Unconfined', 'pil1': '6 µm', 'pil2': '2.6 µm', 'pil3': '1.3 µm'}
disorder_mapping = {'dis0': '0x Disorder', 'dis1': '0x Disorder', 'dis2': '1x Disorder', 
                    'dis3': '2x Disorder', 'dis4':'3x Disorder'}

# Replace the category names in the DataFrame
df['confinement'] = df['confinement'].replace(pillar_mapping)
df['disorder'] = df['disorder'].replace(disorder_mapping) 

#Merge the two columns to one 'condition' column
df['condition'] = df['confinement'] + '; ' + df['disorder']
#df.drop(['confinement', 'disorder'], axis=1, inplace=True)
sub_dfs = {dt: group for dt, group in df.groupby('dt')}

sub_dfs[1]


df_exploded = sub_dfs[1].explode(list('psa'))
df_exploded['p'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded['s'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded['a'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_exploded = df_exploded.rename(columns={'p':'rescaled_pos', 's':'speed', 'a':'angle'})
df_exploded['condition'] = df_exploded['condition'].astype('category')
df_exploded


# ### Define Run Using Speed and Angle Thresholds

average_velocity = df_exploded.groupby(['condition'])['speed'].mean().reset_index()
avg_vel = average_velocity.rename(columns={'speed':'Mean_Speed'})
mean_speed_unconfined = avg_vel.iloc[-1,-1]

half_mean_speed = round(mean_speed_unconfined/2,3)
angle_thresh = round(np.pi/4,3)

print(half_mean_speed)
print(angle_thresh)


# ### Calculate Run Lengths

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
df_run_lengths


# # Chord length analysis (Leone Luzzatto)

# ### Make histograms from the measured swim lengths

def get_histogram(condition, bins):
    runs = df_run_lengths.loc[ df_run_lengths['condition'] == condition ]
    runs = runs['run_length'].to_numpy()
    run_hist, bins = np.histogram( runs, bins=bins, density=True )
    return run_hist


# In[15]:


# Define the bins used in the all histograms in this analysis

# bins = np.logspace( np.log10(1e-1), np.log10(4e2), 40 )
bins = np.linspace(0, 400, 801)

swim_hist_dict = {}
swim_hist_dict['unconfined'] = get_histogram('Unconfined; 0x Disorder', bins)

for confinement, disorder in itt.product( [6, 2.6, 1.3], [0, 1, 2, 3] ):
        condition = f'{confinement} µm; {disorder}x Disorder'
        swim_hist_dict[(confinement, disorder)] = get_histogram(condition, bins)


# ### Measure the chord length distributions

# Function -- generate an image of a disordered environment
# C : confinement
# D : disorder
# d : diameter of the obstacles
# L : linear size of the system
# scale : µm-to-pixel conversion factor

def make_environment(C, D, d, L, scale):
    
    C *= scale
    d *= scale
    L *= scale
    
    env = np.zeros( [L, L], dtype=np.uint8 )
    
    x = np.arange(0, L-2*d-C, C+d ) + d+C/2
    centers = np.array([center for center in itt.product( x, repeat=2 )])
    
    N = len(x)
    displace = np.random.random(size=[N**2,2] ) * C*D - C*D/2
    
    centers += displace

    for center in centers:
        disk = skm.draw.disk(center, d/2)
        env[disk] = 1
    
    return env


# Measure random chords and generate histograms that approximate the chord length distributio in different environments

pad = 0.                        # This quantity can be added to the diameter of the obstacles to account for the finite size of the bacteria
stop = 0.                       # Can be subtracted from the measured chords to account for the finite size of the bacteria
thresh = 0.                     # Minimum length of chords considered in the histogram (measured runs are all > 0.5 µm)
d, L, scale = 10+pad, 400, 10
N_points, N_angles = 400, 36      # Number of random points and number of random directions from each point used to measure chord lengths
frame=0.9                       # Prevents us from picking points too close to the edge of the system, because chords starting from points at the edge of the systems and directed outwards do not encounter any obstacles

chord_length_hist_dict = {}
chord_lengths_list_dict = {}

print('Measuring chord lengths ...')

for C, D in itt.product( [6, 2.6, 1.3], [0,1,2,3] ):

    print('Confinement: {}; Disorder: {} ...'.format(C, D))

    chord_lengths = []
    env = make_environment(C, D, d+pad, L, scale)

    # Identify random starting points for the chords
    points = []
    i = 0
    while i < N_points:
        point_found = False
        while not point_found:
            p = np.random.random(size=[2,])
            p *= (L - 2*d - C) * scale*frame
            p += (d + C/2) * scale*frame
            p += (L - 2*d - C) * scale*(1-frame) / 2
            px, py = p.astype(int)
            point_found = not env[px, py]
        i += 1
        points.append([px, py])
    
    # Pad the environment's size to avoid running into the boundaries during the computation
    new_env = np.zeros( [3*L*scale,3*L*scale], dtype=np.uint8 )
    new_env[L*scale:-L*scale , L*scale:-L*scale] = env
    env = new_env
    points = np.array(points) + L*scale

    # Draw lines from each starting point in random directions and find the first intersection with a wall
    for p in points:
        circle = skm.draw.circle_perimeter(*p, L*scale)
        circle = np.array(circle).transpose()
        circle = np.unique( circle, axis=0 )
        random_idxs = np.random.choice( len(circle), size=[N_angles,] )
        line_endpoints = circle[random_idxs]

        for q in line_endpoints:
            line = skm.draw.line(p[0], p[1], q[0], q[1])
            wall = np.where(env[line]==1)

            if len(wall[0]) == 0:
                chord_lengths.append(399.) # Arbitrary large distance assigned to lines that do not encounter a wall
                continue
            
            r = np.array(line)[:, wall[0][0]]
            dist = np.sqrt(np.sum((p-r)**2)) / scale - stop
            if (dist > thresh):
                chord_lengths.append( dist )
    
    chord_lengths_list_dict[(C,D)] = chord_lengths
    chord_length_hist_dict[(C,D)] = np.histogram( chord_lengths, bins=bins, density=True )[0]

print('Done!')


# ### Compute the conditional distribution from the chord length and swim length distributions

# Function -- Given two histograms (probability distributions), calculate he histogram corresponding
# to the conditioned distribution. Given a first random variable r with probability distribution function
# (p.d.f.) p(r) and cumulative distribution function (c.d.f.) P(r) and a second random variable s with
# p.d.f. q(s) and c.d.f. Q(s), we define a new random variable z as z = min(r,s). Its p.d.f. is given
# by f(z) = p(z)*(1-Q(z)) + (1-P(z))*q(z).

def min_hist(p, q, bins, min=1): # min can be used to set the smallest distance observable in the experiments (0.5 µm)
    weights = np.diff(bins)
    p /= np.dot(p, weights)
    q /= np.dot(q, weights)
    P = np.cumsum( p * weights )
    Q = np.cumsum( q * weights )
    f = p*(1-Q) + (1-P)*q
    for z in range(min):
        f[z]=0
    return f / np.dot(f, weights)

# Compute the conditioned histograms

conditioned_hist_dict = {}

for CD in itt.product([6, 2.6, 1.3], [0, 1, 2, 3]):
    conditioned_hist_dict[CD] = min_hist( swim_hist_dict['unconfined'], chord_length_hist_dict[CD], bins )


# ### Plot the results


# #### C.d.f

mpl.rcParams['font.family'] = 'Arial'

weights = np.diff(bins)

fig, axs = plt.subplots(
    4, 3, figsize=(9,8),
    sharey='row', sharex='col'
)
axs = np.ravel(axs)

colors = [
    'xkcd:cerulean',
    'xkcd:cerulean', 
    'xkcd:orange', 
    'xkcd:orange', 
    'xkcd:frog green',
    'xkcd:frog green', 
    '#000000', # black
    '#b0b0b0'  # light gray
]

plotlist = []

for i, (D, C) in enumerate( itt.product([0,1,2,3], [6, 2.6, 1.3]) ):
    CD = (C, D)
    p0, = axs[i].plot( bins, np.insert(np.cumsum( chord_length_hist_dict[CD]   * weights ) / np.sum( chord_length_hist_dict[CD]   * weights ), 0, 0), '-', color=colors[6], zorder=0       )
    p1, = axs[i].plot( bins, np.insert(np.cumsum( swim_hist_dict['unconfined'] * weights ) / np.sum( swim_hist_dict['unconfined'] * weights ), 0, 0), '-', color=colors[7], zorder=1       )
    p2, = axs[i].plot( bins, np.insert(np.cumsum( swim_hist_dict[CD]           * weights ) / np.sum( swim_hist_dict[CD]           * weights ), 0, 0), '-', color=colors[i%3*2],zorder=2   )
    p3, = axs[i].plot( bins, np.insert(np.cumsum( conditioned_hist_dict[CD]    * weights ) / np.sum( conditioned_hist_dict[CD]    * weights ), 0, 0), '--', color=colors[i%3*2+1]) 
        
    if i < 3:
         plotlist.append([p1,p0,p2,p3]) # Only needed for the legend


# Add titles and axes labels
axs = np.reshape(axs, [4,3])
for (c, C), D in itt.product( enumerate([6, 2.6, 1.3]), [0,1,2,3] ):
        axs[D,c].set_title( f'C = {C} µm, D = {D}' )
for D in range(4):
    axs[D,0].set_ylabel( 'Probability',fontsize=14 )
    axs[D,0].tick_params(axis='y', labelsize=12)
for c in range(3):
    axs[-1,c].set_xlabel( 'Distance (µm)',fontsize=14 )
    axs[-1,c].set_xticks([0,5,10,15,20,25])
    axs[-1,c].tick_params(axis='x', labelsize=12)

# Set axes limits
for ax in axs.ravel():
    ax.set(
        xlim=(0.,25),
        ylim=(0.,1)
    )

# Make a nice legend
legend_tuples = [ tuple( [ plotlist[0][0] ] ) ] + [ tuple( [ plotlist[i][j] for i in range(3) ] ) for j in [2,3] ] + [ tuple( [ plotlist[0][1] ] ) ]
fig.legend(
    legend_tuples,
    ['Unconfined', 'Observed', 'Conditioned', 'Chord length' ],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc='center', bbox_to_anchor=(0.5, -0.01),
    ncols=4, handlelength=4, handletextpad=0.5, labelspacing=0.5,
    fancybox=False, edgecolor='#909090', facecolor='none',
    fontsize='large'
)
#plt.tight_layout()
plt.savefig('fig7-chord_length_analysis.png',transparent=True,dpi=500,bbox_inches='tight')
plt.show()


# ## Compare the distributions

# ### R² score

from sklearn.metrics import r2_score

print('-- R² score comparisons between swim (run) length distribution and other distributions --')
print('\n\t\t\tunconfined distr.\tchord length distr.\tconditioned distr.')
for CD in itt.product([6, 2.6, 1.3], [0,1,2, 3]):
    C, D = CD
    
    print(f'{C} µm; {D}x Disorder\t', end='')
    for f1, f2 in itt.product( [swim_hist_dict[CD]], [ swim_hist_dict['unconfined'], chord_length_hist_dict[CD], conditioned_hist_dict[CD] ] ):    
        
        r2 = r2_score(f1, f2)
        
        print(f'{r2:.6f}\t\t', end='')
    print()


# # ### Wasserstein distance

# # In[226]:


# print('-- Wasserstein distance from the observed distributions --')
# print('\n\t\t\tunconfined distr.\tchord length distr.\tconditioned distr.')
# for CD in itt.product([6, 2.6, 1.3], [0,1,2, 3]):
#     C, D = CD
    
#     print(f'{C} µm; {D}x Disorder\t', end='')
#     for f1, f2 in itt.product( [swim_hist_dict[CD]], [ swim_hist_dict['unconfined'], chord_length_hist_dict[CD], conditioned_hist_dict[CD] ] ):
#             f1 = np.maximum(f1, 0)
#             f2 = np.maximum(f2, 0)
            
#             W_dist = sp.stats.wasserstein_distance(
#                 bins[1:], bins[1:],                               
#                 u_weights=f1,
#                 v_weights=f2
#             )
            
#             print(f'{W_dist:.6f}\t\t', end='')
#     print()


# # ### Kolmogorov-Smirnov test (two-samples)

# # In[242]:


# # Organize the lists of observed swim lengths in a dictionary
# swim_lengths_list_dict = {}

# runs = df_run_lengths.loc[ df_run_lengths['condition'] == 'Unconfined; 0x Disorder' ]
# swim_lengths_list_dict['unconfined'] = runs['run_length'].to_numpy()

# for C, D in itt.product( [6, 2.6, 1.3], [0,1,2,3] ):
#     condition = f'{C} µm; {D}x Disorder'
#     runs = df_run_lengths.loc[ df_run_lengths['condition'] == condition ]
#     swim_lengths_list_dict[(C, D)] = runs['run_length'].to_numpy()

# # draw samples of the conditioned distributions
# minimum_list_dict = {}
# for CD in itt.product( [6, 2.6, 1.3], [0,1,2,3] ):
#     y1 = np.array(swim_lengths_list_dict['unconfined'])
#     y2 = np.array(chord_lengths_list_dict[CD])
#     L = min(len(y1), len(y2))
#     y1 = np.random.choice(y1, size=[5000,])
#     y2 = np.random.choice(y2, size=[5000,])
#     minimum_list_dict[CD] = np.minimum(y1, y2)[ np.where(y2 >= 0.5) ]

# # Perform the two-samples KS
# print('-- Two-samples KS test between the observed distribution --')
# print('\n\t\t\tunconfined distr.\tchord length distr.\tconditioned distr.')
# for CD in itt.product([6, 2.6, 1.3], [0,1,2,3]):
#     C, D = CD
#     print(f'{C} µm; {D}x Disorder\t', end='')
#     for y1, y2 in itt.product( [swim_lengths_list_dict[CD]], [ swim_lengths_list_dict['unconfined'], chord_lengths_list_dict[CD], minimum_list_dict[CD] ] ):
            
#             KS_stat = sp.stats.ks_2samp(y1, y2)
            
#             print(f'{KS_stat.statistic:.4f} (p={KS_stat.pvalue:.2f})\t\t', end='')
#     print()

