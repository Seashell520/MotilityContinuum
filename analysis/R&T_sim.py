#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### This script is for the simulation of the modified run-and-tumble model described in the manuscript. 
### It outputs Figure 6b & 6c as well as Supplmentary Figs 10-12. Written by Jasmine Nirody. Modified by Haibei Zhang.
import re, math, sys, os, random
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import matplotlib as mpl
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob, csv
from scipy.stats import mode
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
from scipy import signal
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from scipy.optimize import curve_fit
import itertools


# In[ ]:


# Parameters
num_bacteria = 200            # Number of bacteria
run_speed = 25.0
simulation_time = 60.              # Total simulation time (seconds)
dt = 0.05                        # Time step for the simulation (seconds)

# Power-law function for fitting
def power_law(t, A, alpha):
    return A * t**alpha
    
def get_consecutive_lengths(sequence):
    tumble_lengths = []
    run_lengths = []
    
    # Initialize the count for the first element
    current_count = 1

    # Iterate through the sequence starting from the second element
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_count += 1
        else:
            # Append to the respective list depending on whether the previous block was ones or zeros
            if sequence[i - 1] == 1:
                tumble_lengths.append(current_count)
            else:
                run_lengths.append(current_count)
            # Reset the count for the new block
            current_count = 1

    # Append the last block
    if sequence[-1] == 1:
        tumble_lengths.append(current_count)
    else:
        run_lengths.append(current_count)

    return tumble_lengths, run_lengths


# In[ ]:


# Function to simulate run-and-tumble dynamics and return ensemble-averaged MSD
def simulate_msd_and_dynamics(tumble_probability, tumble_failure_probability):
    positions = np.zeros((num_bacteria, 2))
    direction = np.random.uniform(0, 2 * np.pi)
    trajectory = np.zeros((num_bacteria, int(simulation_time/dt), 2))
    tumble = np.zeros((num_bacteria,int(simulation_time/dt)),dtype = 'int')
    tumble_lengths = []
    run_lengths = []
    
    for b in range(num_bacteria):
        for i in range(int(simulation_time/dt)):
            if i == 0:
                if np.random.rand() < tumble_probability*np.random.exponential(0.2)*dt:
                    tumble_time = int(np.random.exponential(0.2)/dt) # start with a tumble
                    if tumble_time > dt:
                        tumble[b][i:i+tumble_time] = 1
                    else:
                        tumble[b][i:i+1] = 1
            if i == int(simulation_time/dt)-1:
                tumble[b][i] == tumble[b][i-1]
            else:
                if tumble[b][i] == 1:
                    if tumble[b][i+1] == 0:
                        if np.random.rand() < tumble_failure_probability: # check if tumble fails
                            tumble_time = int(np.random.exponential(0.2)/dt)
                            if tumble_time > dt:
                                if i+1+tumble_time < len(tumble[b])-1:
                                    tumble[b][i+1:i+1+tumble_time] = 1
                                else:
                                    tumble[b][i+1:-1] = 1
                            else:
                                tumble[b][i+1:i+2] = 1
                else:
                    if np.random.rand() < tumble_probability*dt:
                        tumble_time = int(np.random.exponential(0.2)/dt)
                        if tumble_time > dt:
                            if i+tumble_time < len(tumble[b])-1:
                                tumble[b][i:i+tumble_time] = 1
                            else:
                                tumble[b][i:-1] = 1
                        else:
                            tumble[b][i:i+1] = 1
            if tumble[b][i] == 1:
                direction = np.random.uniform(0, 2 * np.pi)
            if tumble[b][i] == 0:
                positions[b,0] += run_speed * np.cos(direction) * dt
                positions[b,1] += run_speed * np.sin(direction) * dt
            trajectory[b][i][0] = positions[b,0]
            trajectory[b][i][1] = positions[b,1]
            
    for b in range(num_bacteria):
        curr_tumble_lengths, curr_run_lengths = get_consecutive_lengths(tumble[b])
        tumble_lengths.extend(curr_tumble_lengths)
        run_lengths.extend(curr_run_lengths)
        #print(sum(tumble))
                    
    #time_lags = np.arange(1, int(simulation_time / dt)*)
    #msd_ensemble_averaged = np.zeros_like(time_lags, dtype=np.float64)
    MSD = [[[] for i in range(int(simulation_time/dt))] for b in range(num_bacteria)]
    mean_MSD = []
    mpl.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(5,4))
    for b in range(num_bacteria):
        x_start = trajectory[b][0][0]
        y_start = trajectory[b][0][1]
        x_pos = []
        y_pos = []
        mean_MSD.append(zeros(len(MSD[b])))
        for i in range(int(simulation_time/dt)):
            x_pos.append(trajectory[b][i][0])
            y_pos.append(trajectory[b][i][1])
        r = [np.sqrt((x-x_start)**2 + (y-y_start)**2) for (x,y) in zip(x_pos,y_pos)]
        
        for time in range(int(len(r))):
            for shift in range(int(len(r))-time):
                displacement_x = (x_pos[time + shift] - x_pos[time])
                displacement_y = (y_pos[time + shift] - y_pos[time])
                MSD[b][shift].extend([(displacement_x+displacement_y)**2])
        for shift in range(len(MSD[b])):
            mean_MSD[b][shift] = mean(MSD[b][shift])
        path = np.array([*trajectory[b]])
        mean_MSD[b] /= mean_MSD[b][1]
        plt.loglog(np.arange(len(mean_MSD[b]))[int(0.05/dt):int(40/dt)] * dt, mean_MSD[b][int(0.05/dt):int(40/dt)],color='lavender',label = 'Individual MSD', linewidth = 1,zorder=0)
    #plt.show()
    print(tumble_probability,tumble_failure_probability)
    meanofmeans_MSD = np.nanmean(np.array(list(itertools.zip_longest(*mean_MSD)),dtype=float),axis=1)
    plt.loglog(np.arange(len(mean_MSD[b]))[int(0.05/dt):int(40/dt)] * dt,meanofmeans_MSD[int(0.05/dt):int(40/dt)],color='dodgerblue', linewidth = 3, label = 'Mean of MSDs',zorder=1)
    plt.loglog(np.arange(len(mean_MSD[b]))[int(0.05/dt):int(0.5/dt)] * dt, power_law(np.arange(len(mean_MSD[b]))[int(0.05/dt):int(0.5/dt)] * dt, 400, 2), '--', color='black',lw='1')
    medium_time_lags = np.arange(len(mean_MSD[b]))[(np.arange(len(mean_MSD[b])) * dt >= 2) & (np.arange(len(mean_MSD[b])) * dt <= 20)] * dt
        #if len(medium_time_lags) > 0:
           # msd_medium = msd_ensemble_averaged[(time_lags >= int(5 / dt)) & (time_lags <= int(25 / dt))]
            #if len(msd_medium) > 1:  # Ensure there's enough data for fitting
                #popt_medium, _ = curve_fit(power_law, medium_time_lags, msd_medium)
    plt.loglog(medium_time_lags, power_law(medium_time_lags, 40, 1), '--',color='black', lw='1')
        #squared_displacements = np.sum(displacements**2, axis=2)
        #msd_ensemble_averaged[lag - 1] = np.mean(squared_displacements)
    #plt.loglog(meanofmeans_MSD)
    plt.xlabel('Lag time τ (s)',fontsize=16)
    plt.ylabel('MSD (µm²)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'Tumble failure = {tumble_failure_probability}; Tumble freq = {tumble_probability}',fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize=14)
    plt.tight_layout()
    #plt.savefig('TP' + str(tumble_probability) + '_TF' + str(tumble_failure_probability) + '_MSD.pdf',dpi=300,transparent=True)
    
    # Normalize the MSD by the first value
    meanofmeans_MSD /= meanofmeans_MSD[1]
    
    #print(tumble)
    #print(trajectory)
   # plt.plot(trajectory[:,0,0],trajectory[:,0,1])
   # plt.show()
    
    return meanofmeans_MSD, np.arange(len(meanofmeans_MSD)), tumble_lengths, run_lengths, fig


# In[ ]:


# Function to plot MSDs for varying tumble frequency or tumble failure probability.
# Inputs are a single value of tumble failure probability (tf) and a list of tumble probability (tp). 
# Outputs are three plots: 1) eMSD plot of fixed tumble failure probability with varying tumble probabilities; 
# 2) a plot of four subplots including time- and ensemble- averaged MSDs; 3) a plot of 16 1-CDF subplots including run and tumble durations 

def plot_msd(tf,tp):
    
    #inset_ax1 = ax.inset_axes([0.05, 0.65, 0.3, 0.3], xlim=[0.1, 0.55], ylim = [3,150], xticklabels = [])
   # inset_ax2 = ax.inset_axes([0.65, 0.05, 0.3, 0.3], xlim=[2, 30], ylim = [150,50000])

    all_run_lengths = []
    all_tumble_lengths = []
    figures = []
    conditions = [(tp[0],tf),(tp[1],tf),(tp[2],tf),(tp[3],tf)]
    #conditions = [(4.0,0.0),(4.0,0.9),(4.0,0.99),(4.0,0.999)]
    all_msd_ensemble_averaged = [[] for i in range(len(conditions))]
    all_time_lags = [[] for i in range(len(conditions))]
    if abs(conditions[0][0] - conditions[1][0]) < 0.01:
        vary = 'failure'
        constant = 'frequency'
    else:
        vary = 'frequency'
        constant = 'failure'
    if vary == 'failure':
        plot_add = 'vary' + vary + '_' + constant + str(conditions[0][0])
    else:
        plot_add = 'vary' + vary + '_' + constant + str(conditions[0][1])
    
    for i in range(len(conditions)):
        tumble_probability = conditions[i][0]
        tumble_failure_probability = conditions[i][1]
        
        all_msd_ensemble_averaged[i], all_time_lags[i], tumble_lengths, run_lengths, subfig = simulate_msd_and_dynamics(tumble_probability, tumble_failure_probability)
        all_tumble_lengths.append(tumble_lengths)
        all_run_lengths.append(run_lengths)
        figures.append(subfig)
    rows = 1
    cols = 4
    figs, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    for i, ax in enumerate(axs.flatten()):
        fig_i = figures[i]

        # Extract data from the original figure
        ax_i = fig_i.axes[0]

        # Set log scale if used in the original plot
        if ax_i.get_xscale() == 'log':
            ax.set_xscale('log')
        if ax_i.get_yscale() == 'log':
            ax.set_yscale('log')
    
        # Copy axis labels and title
        ax.set_xlabel(ax_i.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax_i.get_ylabel(), fontsize=16)
        ax.set_title(ax_i.get_title(), fontsize=15)
        # Set tick parameters
        ax.tick_params(labelsize=16)
        
       # Recreate the plot by extracting the data
        for line in ax_i.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            line_width = line.get_linewidth()  # Get the original line width
            line_style = line.get_linestyle()  # Get the original line style
            ax.plot(x_data, y_data, label=line.get_label(), color=line.get_color(), linewidth=line_width, linestyle=line_style)
        
         # Copy the legend
        handles, labels = ax_i.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=15)

    plt.tight_layout()
    plt.savefig(f'MSD_tf_{tf}.pdf',dpi=300,transparent=True)
    plt.show()
    plt.close()
    # MSD Plot
    #for axes in ax, inset_ax1, inset_ax2:
    mpl.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots()
    colors = ['blue','darkorange','green','deeppink','gold','indigo']
    for i in range(len(conditions)):
        tumble_probability = conditions[i][0]
        tumble_failure_probability = conditions[i][1]
        msd_ensemble_averaged = all_msd_ensemble_averaged[i]
        time_lags = all_time_lags[i]
        if vary == 'frequency':
            lab = str(tumble_probability)
        else:
            lab = str(tumble_failure_probability)
        ax.loglog(time_lags[int(0.05/dt):int(40/dt)] * dt, msd_ensemble_averaged[int(0.05/dt):int(40/dt)], label=lab, color=colors[i])

            # Fit power law to the first 0.1 seconds
        short_time_lags = time_lags[(time_lags * dt > 0.05) & (time_lags * dt < 0.5)] * dt
        #msd_short = msd_ensemble_averaged[time_lags * dt < 1]
        #if len(msd_short) > 1:  # Ensure there's enough data for fitting
            #popt, _ = curve_fit(power_law, short_time_lags, msd_short)
        ax.loglog(time_lags[int(0.05/dt):int(0.5/dt)] * dt, power_law(time_lags[int(0.05/dt):int(0.5/dt)] * dt, 400, 2), '--', color='black',lw='1')
        #ax.loglog(time_lags[int(0.05/dt):int(0.5/dt)] * dt, power_law(time_lags[int(0.05/dt):int(0.5/dt)] * dt, 80, 1.5), '--',color='black', lw='1')

        # Fit power law to medium time lags
        medium_time_lags = time_lags[(time_lags * dt >= 2) & (time_lags * dt <= 20)] * dt
        #if len(medium_time_lags) > 0:
           # msd_medium = msd_ensemble_averaged[(time_lags >= int(5 / dt)) & (time_lags <= int(25 / dt))]
            #if len(msd_medium) > 1:  # Ensure there's enough data for fitting
                #popt_medium, _ = curve_fit(power_law, medium_time_lags, msd_medium)
        ax.loglog(medium_time_lags, power_law(medium_time_lags, 40, 1), '--',color='black', lw='1')

    ax.set_xlim([0.04,60])
    ax.set_ylim([0.5,100000])
    #ax.set_title('Normalized Ensemble-Averaged MSD for Varying Tumble Probability')
    ax.set_xlabel('Time (s)',fontsize=16)
    ax.set_ylabel('Normalized MSD',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.85])

    # Put a legend below current axis
    if vary == 'frequency':
        ax.legend(title = 'Tumble ' + constant + '=' + str(tumble_failure_probability) + ', Tumble ' + vary + '=', loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=len(conditions), fontsize=12)
    if vary == 'failure':
        ax.legend(title = 'Tumble ' + constant + '=' + str(tumble_probability) + ', Tumble ' + vary + '= :', loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=len(conditions), fontsize=12)

#    _, c = ax.indicate_inset_zoom(inset_ax1, edgecolor="black")
#    c[0].set_visible(False)
#    c[1].set_visible(False)
#    c[2].set_visible(False)
#    c[3].set_visible(False)
#    _, c = ax.indicate_inset_zoom(inset_ax2, edgecolor="black")
#    c[0].set_visible(False)
#    c[1].set_visible(False)
#    c[2].set_visible(False)
#    c[3].set_visible(False)
    #ax.legend(loc='lower right', fontsize=10)
    
#    else:
#        tumble_probability = 3.0                       # Fixed tumble probability
#        tumble_failure_probabilities = [0.0, 0.5, 0.9, 0.99, 0.999]  # Updated varying tumble failure probabilities
#        for tumble_failure_probability in tumble_failure_probabilities:
#            msd_ensemble_averaged, time_lags, tumble_lengths, run_lengths = simulate_msd_and_dynamics(tumble_probability, tumble_failure_probability)
#            all_tumble_lengths.append(tumble_lengths)
#            all_run_lengths.append(run_lengths)
#
#            # MSD Plot
#            #for axes in ax: #, inset_ax1, inset_ax2:
#            ax.plot(time_lags * dt, msd_ensemble_averaged, label=f'Tumble Failure Prob = {tumble_failure_probability}')
#
##            # Fit power law to the first 0.1 seconds
#            short_time_lags = time_lags[time_lags*dt < 0.5]
#            msd_short = msd_ensemble_averaged[:len(short_time_lags)]
#            if len(msd_short) > 1:  # Ensure there's enough data for fitting
#                popt, _ = curve_fit(power_law, short_time_lags, msd_short)
#                print(popt)
##                inset_ax1.plot(short_time_lags, power_law(short_time_lags, *popt), '--',
##                        label=f'Fit (Tumble Failure Prob = {tumble_failure_probability}, α = {popt[1]:.2f})')
##
##            # Fit power law to medium time lags
##            medium_time_lags = time_lags[(time_lags * dt >= 3) & (time_lags * dt <= 20)] * dt
##            if len(medium_time_lags) > 0:
##                msd_medium = msd_ensemble_averaged[(time_lags >= int(3 / dt)) & (time_lags <= int(20 / dt))]
##                if len(msd_medium) > 1:  # Ensure there's enough data for fitting
##                    popt_medium, _ = curve_fit(power_law, medium_time_lags, msd_medium)
##                    inset_ax2.plot(medium_time_lags, power_law(medium_time_lags, *popt_medium), '--',
##                            label=f'Fit (Tumble Prob = {tumble_probability}, α (medium) = {popt_medium[1]:.2f})')
#
#        ax.set_xscale('log')
#        ax.set_yscale('log')
#        #ax.set_title('Normalized Ensemble-Averaged MSD for Varying Tumble Failure Probability')
#        ax.set_xlabel('Time (s)')
#        ax.set_ylabel('Normalized MSD')
#        ax.legend(loc='lower right', fontsize=10)

    # Inset for short time lags
   # inset_ax1.set_xscale('log')
   # inset_ax1.set_yscale('log')
 #   inset_ax1.set_xticklabels([])
  #  inset_ax1.set_yticklabels([])
   # inset_ax2.set_xscale('log')
   # inset_ax2.set_yscale('log')
   # inset_ax2.set_xticklabels([])
    #inset_ax2.set_yticklabels([])
    plt.savefig('MSD_' + plot_add + '.pdf', dpi = 300, transparent = True)
    plt.close()

   # if vary_tumble_frequency:
   #     plt.savefig('MSD_vary_tumble_frequency.pdf')
   # else:
   #     plt.savefig('MSD_vary_tumble_failure.pdf')
    mpl.rcParams['font.family'] = 'Arial'
    plt.subplots(rows,cols,figsize=(cols*4, rows*4.2))
    ticks = 0
    for i in range(len(all_tumble_lengths)):
        ticks += 1
        ax = plt.subplot(rows,cols,ticks)
        #if vary_tumble_frequency:
        plot_tumble = [tl*dt for tl in all_tumble_lengths[i]]
        sns.ecdfplot(ax=ax,data=plot_tumble, complementary=True, color='xkcd:barney purple', label=r'$<T_{tumble}> =$ ' + str(round(mean(all_tumble_lengths[i])*dt,2)) + ' s')
        #axes[i].set_ylabel('Tumble time count')
        #axes[i].legend()
        print('Mean tumble: ', mean(all_tumble_lengths[i])*dt)
        #plt.show()
        plot_run = [rl*dt for rl in all_run_lengths[i]]
        sns.ecdfplot(ax=ax, data=plot_run, complementary=True, color='dodgerblue', label=r'$<T_{run}> =$ ' + str(round(mean(all_run_lengths[i])*dt,2)) + ' s')
        print('Mean run: ', mean(all_run_lengths[i])*dt)
        ax.set_ylabel('')
        ax.legend(prop={'size': 12})
        ax.set_box_aspect(1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0.03,65)
        ax.set_ylim(1e-4,1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Probability',fontsize=16)
        ax.set_xlabel('Duration (s)',fontsize=16)
        ax.set_title(f'Tumble failure = {tf} \n Tumble freq = {tp[ticks-1]}',fontsize=14)
    plt.tight_layout()
        #plt.show()
#        else:
#            tp = str(tumble_probability)
#            tf = str(tumble_failure_probabilities[i])
#            print(tumble_probability,tumble_failure_probabilities[i])
#            sns.kdeplot(ax=ax1,data=all_tumble_lengths[i], label=tumble_failure_probabilities[i])
#            ax1.set_ylabel('Tumble time count')
#            ax1.legend()
#            print('Mean tumble: ', mean(all_tumble_lengths[i])*dt)
#            #plt.show()
#            sns.kdeplot(ax=ax2, data=all_run_lengths[i], label=tumble_failure_probabilities[i])
#            print('Mean run: ', mean(all_run_lengths[i])*dt)
#            ax2.set_ylabel('Run time count')
#            ax2.legend()
#            #plt.show()
    plt.savefig('runtumbles_' + plot_add + '.pdf',dpi=300,transparent=True)

#Example usage:
#tf = 0
#tp = [1.0,2.0,4.0,8.0]
#plot_msd(tf,tp)


# In[ ]:


# Simulate under the fixed tumble failure probability of 0.0 with varying tumble probability
tf = 0.0
tp = [1.0, 2.0, 4.0, 8.0]
plot_msd(tf,tp)


# In[ ]:


# Simulate under the fixed tumble failure probability of 0.0 with varying tumble probability
tf_9 = 0.9
plot_msd(tf_9,tp)


# In[ ]:


# Simulate under the fixed tumble failure probability of 0.0 with varying tumble probability
tf_99 = 0.99
plot_msd(tf_99,tp)


# In[ ]:


tf_999 = 0.999
plot_msd(tf_999,tp)


# In[ ]:




