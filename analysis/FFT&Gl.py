#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

##### The script is to compute global angle distributions and fast fourier transformation of trajectories -
##### for figure 5 and supplementary figure 9
import os
import glob,csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import random
from PIL import Image
from skimage.io import imread


### Define a function to calculate the arc tangent value from 0-2pi
def myatan(y, x=None):
    if x is None:  # If only y is given, set x to 1
        x = 1

    v = np.nan
    if x > 0:
        v = np.arctan(y / x)
    if y >= 0 and x < 0:
        v = np.pi + np.arctan(y / x)
    if y < 0 and x < 0:
        v = -np.pi + np.arctan(y / x)
    if y > 0 and x == 0:
        v = np.pi / 2
    if y < 0 and x == 0:
        v = -np.pi / 2
    if v < 0:
        v += 2 * np.pi

    return v #returns the radian

# Example usage
# y = 1
# x = 1
# angle = myatan(y, x)
#print(angle) 

### Global angle computation
def anglestart(x_pos,y_pos,x_start,y_start,pixel_to_micron,skip):
    angles = []
    for t in range(0,len(x_pos),skip):
        x_repos = pixel_to_micron*(x_pos[t]-x_start)
        y_repos = pixel_to_micron*(y_pos[t]-y_start)
        angle = myatan(y_repos,x_repos)
        angles.append(angle)
    return angles

### define sliding window algorithm 
def sliding_window(elements, window_size, overlap):
    means = []
    step = window_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than window size")
    if len(elements) <= window_size:
        return elements
    for i in range(0, len(elements) - window_size +1,step):
        mean = np.mean(elements[i:i+window_size])
        means.append(mean)
        #print((elements[i:i+window_size]),'k')
    return means

# Example usage
# lst = [1, 1, 1, 2, 2, 2, 3, 3] 
# window_size = 4
# overlap = 2

# result = sliding_window(lst, window_size, overlap)
# print(result)

## fetch the files 
upperdir = '/'.join(os.getcwd().split('/')[:-1]) #define upper directory
lowerdir = '/sorted_tracks/' #define lower directory where all the data is located
files = glob.glob(upperdir + lowerdir + '*.csv') #grab all the files in their pathways 
files = sorted(files) #sort the files based on the naming
len(files)

# Construct a dictionary for data storage and analysis
datadic = {} #set up a data directory
dev = [] #number of device
pil= [] #number of confinement
dis = [] #number of disorders
rep = [] #number of repititions 
pixel_to_micron = 0.656 #pixel to micron (from Miles)
fps = 20 #frames per second
skip = 1
window_size =5
overlap = 4

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
        #datadic[device][pillar][disorder]['MSD'] = [[]]
        datadic[device][pillar][disorder]['reposition'] = []
        datadic[device][pillar][disorder]['angle_start'] = []
        
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
        datadic[device][pillar][disorder]['reposition'].append([(pixel_to_micron*(x_pos[t]-x_start), pixel_to_micron*(y_pos[t]-y_start)) for t in range(len(x_pos))])
        
        #compute the angles with respect to the same origin (0,0)
        angles = anglestart(x_pos,y_pos,x_start,y_start,pixel_to_micron,skip)[1:]
        angle_sl = sliding_window(angles,window_size,overlap)
        datadic[device][pillar][disorder]['angle_start'].append(angle_sl)
        
    #     Test whether the trajecories are stored properly           
#     if device == 'dev1' and pillar == 'pil1' and disorder == 'dis1':
#          for i in range(len(datadic[device][pillar][disorder]['reposition'])):
#             path = np.array(datadic[device][pillar][disorder]['reposition'][i])
#             plt.plot(*path.T) # ".T" attribute is used to transpose the array, swapping its rows and columns. # "*" unpacks the rows of the transposed array
#             plt.xlabel('µm')
#             plt.ylabel('µm')

#     plt.show()

# Store all global angles in the corresponding lists 
alpha = []

for dev in datadic.keys():
    for pil in datadic[dev].keys():
        for dis in datadic[dev][pil].keys():
            alpha.extend([(pil, dis, angle) for angle in datadic[dev][pil][dis]['angle_start']])

df = pd.DataFrame(alpha,columns=['confinement', 'disorder', 'alpha'])

pillar_mapping = {'pil0': 'Unc', 'pil1': 'C = 6 µm', 'pil2': 'C = 2.6 µm', 'pil3': 'C = 1.3 µm'}
disorder_mapping = {'dis0': 'no pillar', 'dis1': 'D = 0', 'dis2': 'D = 1', 
                    'dis3': 'D = 2', 'dis4':'D = 3'}

# Replace the category names in the DataFrame
df['confinement'] = df['confinement'].replace(pillar_mapping)
df['disorder'] = df['disorder'].replace(disorder_mapping) 

# Merge the two columns to one 'condition' column
df['condition'] = df['confinement'] + '; ' + df['disorder']

df_exploded = df.explode('alpha')

# All the possible conditions
condition = ['Unc; no pillar', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C = 6 µm; D = 2', 'C = 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

# Plot the polar diagrams of global angles
fig, axes = plt.subplots(nrows=4, ncols=4, subplot_kw=dict(projection="polar"), figsize=(20, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, ax in enumerate(axes.flat):
    if i < len(condition):
        subset_df = df_exploded[df_exploded['condition'] == condition[i]]['alpha']
        #print(len(subset_df))
        # Define binning
        bins = np.linspace(0, 2*np.pi, 37)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(subset_df, bins=bins)
        
        # Normalize histogram
        hist = hist / (len(subset_df))
        
        # Wrap around the histogram
        hist = np.concatenate((hist, [hist[0]]))
        bin_edges = np.concatenate((bin_edges, [bin_edges[-1] + (bin_edges[1] - bin_edges[0])]))
        
        #ax.plot(bin_edges, np.concatenate((hist, [hist[0]])), color='blue', zorder=2)
        
        # Create the bar plot on the polar axis
        bars = ax.bar(bin_edges, np.concatenate((hist, [hist[0]])), width=(2*np.pi)/36, align='edge')
        ax.set_theta_zero_location("E")
        
        # Use custom colors and opacity
        for r, bar in zip(bin_edges[:-1], bars):
            bar.set_facecolor('xkcd:azure')
            bar.set_edgecolor('black')
            bar.set_alpha(0.8)
            bar.set_zorder(2)
        
        # Change the size of the degree labels
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_yticks([0.02,0.04,0.06])
        # Set grid style
        ax.grid(True, linestyle='dashed', zorder=1)
        ax.set_title(condition[i], fontsize=20,color='xkcd:dusk blue',weight='bold')
    else:
        # Hide any extra subplots if the number of conditions is less than 16
        ax.axis('off')
#fig.suptitle('Polar Histograms of Directional Angle (Sliding window: 5 frames with 80% overlap)',fontsize=20)
fig.subplots_adjust(top=0.93)
plt.rcParams['font.family'] = 'Arial'
plt.savefig('fig5c_suppfig9_polar_angle.png',dpi=300,transparent=True)
#plt.show()
plt.close()


##### FFT analysis 
conditions = ['Unc', 'C = 6 µm; D = 0', 'C = 6 µm; D = 1', 'C= 6 µm; D = 2', 'C= 6 µm; D = 3',
       'C = 2.6 µm; D = 0', 'C = 2.6 µm; D = 1', 'C = 2.6 µm; D = 2', 'C = 2.6 µm; D = 3', 
        'C = 1.3 µm; D = 0', 'C = 1.3 µm; D = 1', 'C = 1.3 µm; D = 2', 'C = 1.3 µm; D = 3']

name = ['Unc', '6-0x', '6-1x', '6-2x', '6-3x',
       '2-0x', '2-1x', '2-2x', '2-3x', 
        '1-0x', '1-1x', '1-2x', '1-3x']

# Initialize lists to store figures and filenames
figures = []
fft_figures = []
fft_filenames = []

#Starting from unconfined -> 6 -> 2.6 -> 1.3
for dev in list(datadic):
    count = 0
    for pil in list(datadic[dev]):
        for dis in list(datadic[dev][pil]):

            total_trajectories = len(datadic[dev][pil][dis]['reposition'])

            random.seed(2008)
            # Randomly select 50 unique indices
            random_indices = random.sample(range(total_trajectories), 60)
                
            fig, ax = plt.subplots(figsize=(6,6))
            # Plot the selected trajectories
            for i in random_indices:
                path = np.array(datadic[dev][pil][dis]['reposition'][i])
                ax.plot(*path.T, linewidth=0.6, c='k') # ".T" transposes the array, "*" unpacks the rows of the transposed array

            plt.xlim(-350, 350)
            plt.ylim(-350, 350)
            plt.axis('off')

            # Store the figure
            figures.append(fig)
            
            # Save the plot temporarily 
            temp_filename = f'{name[count]}-repos.png'
            plt.savefig(temp_filename,dpi=300,bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Convert the saved image to grayscale and inverse binarize it
            # Open the saved image
            img = Image.open(temp_filename).convert('L')

            # Define a threshold value for inverse binarization
            threshold = 128

            # Apply the inverse binarization
            inversely_binarized_img = img.point(lambda p: p > threshold and 0)  # Set pixels above threshold to 0 (black)
            inversely_binarized_img = img.point(lambda p: p < threshold and 255)  # Set pixels above threshold to 0 (black)
            #inversely_binarized_img = inversely_binarized_img.point(lambda p: p == 0 and 255)  # Set black pixels to white
            #inversely_binarized_img = inversely_binarized_img.point(lambda p: p == 255 and 0)  # Set white pixels to black

            # Save the inversely binarized image
            inverse_bin_filename = f'{name[count]}-repos-inversely-binarized.png'
            inversely_binarized_img.save(inverse_bin_filename)
            plt.close()
            
            # FFT and plot the magnitude spectrum
            img = imread(inverse_bin_filename)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20*np.log(np.abs(fshift))
            
            # Create the FFT figure
            fft_fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title(f'{condition[count]}', fontsize=26)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            axs[1].imshow(magnitude_spectrum, cmap='seismic')
            axs[1].set_title(f'{condition[count]}', fontsize=26)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
                  
            plt.savefig(f'fig5ab_suppfig9_{name[count]}_fft.png',dpi=300)
            #plt.show()
            plt.close(fft_fig)
            
            # Delete the temporary images after processing
            os.remove(temp_filename)
            os.remove(inverse_bin_filename)

            count += 1


