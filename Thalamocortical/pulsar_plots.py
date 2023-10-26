#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from matplotlib.pyplot import cm


# In[ ]:

all_trialtypes = {'correct responses': ['rewarded hit', 'unrewarded hit'],
                 'incorrect responses': ['rewarded FA', 'unrewarded FA'],
                 'unexpected omission': ['unrewarded hit', 'unrewarded hit'],
                 'expected omission': ['unrewarded FA', 'unrewarded FA'],
                 'expected reward': ['rewarded hit', 'rewarded hit'],
                 'unexpected reward': ['rewarded FA', 'rewarded FA'],
                 'unexpected outcomes': ['unrewarded hit', 'rewarded FA'],
                 'expected outcomes': ['unrewarded FA', 'rewarded hit'],
                 'responses':['rewarded hit','rewarded FA','unrewarded hit','unrewarded FA'],
                 'no responses':['unrewarded miss','unrewarded CR'],
                 'rewarded responses':['rewarded hit','rewarded FA'],
                 'unrewarded responses':['unrewarded hit','unrewarded FA']}

modulation_trialtypes = {'correct responses':['hit','FA'],
              'incorrect responses':['FA','hit'],
              'unexpected omission':['unrewarded hit','unrewarded FA'],
              'expected omission':['unrewarded FA','unrewarded hit'],
              'expected reward':['rewarded hit','rewarded FA'],
              'unexpected reward':['rewarded FA','rewarded hit'],
              'Go':['Go', 'NoGo'],
              'NoGo':['NoGo', 'Go'],
              'responses':['hit', 'FA'],
              'no responses':['CR','miss'],
              'rewarded':['rewarded', 'unrewarded'],
              'unrewarded':['unrewarded', 'rewarded'],
              'unexpected outcomes':['unrewarded hit','rewarded FA'],
              'expected outcomes':['unrewarded FA','rewarded hit']
             }

boxplot_colors = {'unexpected omission':'magenta',
          'expected omission':'grey',
          'expected reward':'navy',
          'unexpected reward':'lightblue',
          'Hit':'green',
          'correct responses':'green',
          'Miss':'grey',
          'FA':'darkred',
          'incorrect responses':'darkred',
          'CR':'darkorange',
          'expected outcomes': 'grey',
          'unexpected outcomes': 'magenta',
          'Go':'navy',
          'NoGo':'grey',
          'rewarded responses':'blue',
          'unrewarded responses':'darkorange',
          'responses':'darkcyan',
          'no responses':'olive'}

trace_colors = {'unexpected omission':'magenta',
          'expected omission':'peru',
          'expected reward':'navy',
          'unexpected reward':'lightblue',
          'correct responses':'green',
          'Miss':'grey',
          'incorrect responses':'darkred',
          'Correct Rejections':'darkorange',
          'expected outcomes': 'black',
          'unexpected outcomes': 'magenta',
          'Go':'navy',
          'NoGo':'grey',
          'rewarded responses':'blue',
          'unrewarded responses':'darkorange',
          'responses':'darkcyan',
          'no responses':'olive'}

task_phases = {'Ndet':'Novice association',
               'Edet':'Expert association',
               'Ndis':'Novice pre-reversal',
               'Edis':'Expert pre-reversal',
               'Nrev':'Novice post-reversal',
               'Erev':'Expert post-reversal'}

def filter_ROIs(cohort_epochs, epoch, ROIs, correlated_ROI_list):
    all_deltaF = pd.concat(cohort_epochs['deltaF'][epoch])
    all_events = pd.concat(cohort_epochs['events'][epoch])

    if ROIs == 'all':
        return all_deltaF, all_events
    else:
        filtered_deltaF = all_deltaF[all_deltaF.index.get_level_values('Unique_ROI').isin(correlated_ROI_list)]
        filtered_events = all_events[all_events['Unique_ROI'].isin(correlated_ROI_list)]
        return filtered_deltaF, filtered_events

def plot_all_pulsar_traces(sessions, seeds, protocols, trialtypes, channels, max_trials, min_trials, squish_factor, y_shrinkage, x_range, scaling_dict, sort_dict, deltaF_ROIs, events_ROIs, cohort_epochs, correlated_ROI_list, ROIs, save):

    if sessions == 'all':
        sessions = deltaF_ROIs.index.get_level_values('Unique session').unique()
        seeds = [0]*len(sessions)
    else:
        sessions = sessions
        seeds = seeds
    for sesh, seed in zip(sessions,seeds):
        if seed in [9,46]:
            tmax = 50
        else:
            tmax = max_trials
        np.random.seed(seed)
        sesh_df = deltaF_ROIs[deltaF_ROIs.index.get_level_values('Unique session') == sesh]
        for channel in channels:
            compartment = 'dendrites' if channel == 'Red' else 'axons'
            if channel in sesh_df.index.get_level_values('Channel'):
                channel_df = sesh_df[sesh_df.index.get_level_values('Channel') == channel]
                for protocol in protocols:
                    protocol_name = task_phases[protocol]
                    if protocol in channel_df.index.get_level_values('Protocol'):
                        protocol_df = channel_df[channel_df.index.get_level_values('Protocol') == protocol]

                        ## Get framerate and corresponding start/end frames ##
                        fps = 30.54 if protocol_df.dropna(axis=1).shape[0] == 367 else 30.3
                        start = math.floor(fps*x_range[0])
                        end = math.floor(fps*x_range[1])

                        for trialtype in trialtypes:                           
                            ## Get events and deltaF for current trial type of interest ##
                            epoch_deltaF, epoch_events = filter_ROIs(cohort_epochs, 'reward', ROIs, correlated_ROI_list) if ('reward' in trialtype) | ('omission' in trialtype) | ('outcomes' in trialtype) else filter_ROIs(cohort_epochs, 'response', ROIs, correlated_ROI_list)
                            epoch_events = epoch_events[(epoch_events['Unique session'] == sesh) & (epoch_events['Channel'] == channel)]
                            ttype_df, ttype_events = extract_ttype_df(protocol_df, epoch_events, trialtype, all_trialtypes)

                            ## Bit of data cleaning ##
                            ttype_df = ttype_df.iloc[:,2:-2] #remove shutter artifact
                            ttype_df = ttype_df[ttype_df.std(axis=1) != 0] #remove no fluorescence trials
                            ttype_df[ttype_df.mean(axis=1) > -0.8] #remove no fluorescence trials
                            ttype_df = ttype_df.loc[:, start:end] #select frames for the desired view range

                            ## Determine how many trials we will plot (one trial per subplot) ##
                            ntrials = len(ttype_df.index.get_level_values('Unique_trial').unique())
                            if ntrials > (min_trials-1):
                                final_trials = tmax if tmax <= ntrials else ntrials

                                ## Randomly sample trials ##
                                subindex = ttype_df.index.get_level_values('Unique_trial').unique()
                                sample_ids = np.random.choice(subindex, final_trials, replace=False)
                                ttype_df = ttype_df[ttype_df.index.get_level_values('Unique_trial').isin(sample_ids)]

                                ## Set figure parameters using set_figure() function ##
                                n_lines = len(ttype_df.index.get_level_values('Unique_ROI').unique()) #number of lines in each subplot
                                if n_lines > 0:
                                    x = [int(x) for x in ttype_df.columns] #the x-values of the lines
                                    if scaling_dict == {}:
                                        fig, ax = set_figure(n_lines, final_trials, squish_factor)
                                    else:
                                        fig, ax = set_figure(n_lines, final_trials, squish_factor,
                                                             xscaling = scaling_dict['xscaling'], yscaling = scaling_dict['yscaling'])

                                    ## Loop through each trial of current trial type dataframe (to plot every ROI for each trial) ##
                                    for col, (trial) in enumerate(ttype_df.index.get_level_values('Unique_trial').unique()):
                                        tnum = trial.split('_')[-1] #current trial number
                                        trial_df = ttype_df[ttype_df.index.get_level_values('Unique_trial') == trial] #single trial deltaF (of all ROIs)
                                        outcome = trial_df.index.get_level_values('Final Outcome')[0] #outcome of current trial number
                                        response_latency = trial_df.index.get_level_values('Response latency')[0]
                                        
                                        ## Sort rows of trial_df using sort_deltaF_rows function ##
                                        if sort_dict == {}:
                                            final_data, non_event_indexes, event_indexes = sort_deltaF_rows(trial, channel, 
                                                                                                            trial_df, events_ROIs, ttype_events)
                                        else:
                                            final_data, non_event_indexes, event_indexes = sort_deltaF_rows(trial, channel, trial_df, 
                                                                                                            events_ROIs, ttype_events,
                                                                                                            sort_dict['sort_by'], 
                                                                                                            sort_dict['ascending'], 
                                                                                                            sort_dict['events_first'])
                                        ## Set index-based line plotting parameters ##
                                        idx_colors = [trace_colors[trialtype] if x in event_indexes else 'darkgrey' for x in range(final_data.shape[0])]
                                        idx_alpha = [1 if x in event_indexes else 1 for x in range(final_data.shape[0])]

                                        ## Plot the resulting pulsar charts onto current axes object (ax[col]) ##
                                        ax_obj = ax[col] if final_trials > 1 else ax
                                        pulsar_plot(final_data, n_lines, x, ax_obj, 
                                                    squish_factor=squish_factor, y_shrinkage=y_shrinkage, 
                                                    axis='off', idx_colors=idx_colors, idx_alpha=idx_alpha)
                                        plot_behavioral_vlines(outcome, trialtype, response_latency, 
                                                               ax_obj, x_range, fps) # plot stimulus, lick, reward times

                                    ## Section for titles, subtitles, and figure saving ##
                                        ax_obj.set_title(f'Trial {tnum}', fontsize=8, y=0, pad=-1)
                                    fig.suptitle(f'{sesh}_{compartment} (seed {seed})',fontsize=10)#\n {compartment} {protocol_name} {trialtype[:-1]} trials', fontsize=10, y=0.98)
                                    fig.savefig(os.path.join(data_directory,f'WATERFALL_{sesh}_{compartment}_{trialtype}_correlated_ROIs.svg')) if save == True else None
                                    plt.show()


# In[2]:


def pulsar_plot(final_data, n_lines, xvalues, ax_obj, squish_factor=3, y_shrinkage=2, axis='off', idx_colors='darkgrey', idx_alpha=1):
    line_mins = []
    for i, (index, rows) in enumerate(final_data.iterrows()):
        line = rows/y_shrinkage + (n_lines - i/squish_factor) #re-scale and re-position line

        ## Store minimum and maximum line values for re-adjusting y-axis limits later ##
        line_mins.append(line.min())
        
    for i, (index, rows) in enumerate(final_data.iterrows()):
        ## Recover line parameters to plot ##
        color = idx_colors[i] if idx_colors != 'darkgrey' else 'darkgrey'
        alpha = idx_alpha[i] if idx_alpha != 1 else 1
        line = rows/y_shrinkage + (n_lines - i/squish_factor) #re-scale and re-position line

        ## Plot resulting line
        if i > 0:
            ax_obj.fill_between(xvalues, line, min(line_mins), lw=1, edgecolor='white', facecolor='white', zorder=i/n_lines)
        ax_obj.plot(xvalues, line, lw=1, c=color, alpha=alpha, zorder=i/n_lines)   
    ax_obj.axis('off')


# In[3]:


def set_figure(n_lines, ncols, squish_factor, wspace=0.1, sharey=True, xscaling=1, yscaling=1):
    x_size = ((0.15)*ncols)*5*xscaling
    y_size = ((n_lines*0.05)/squish_factor)*5*yscaling
    
    if ncols > 1:
        fig, ax = plt.subplots(nrows=1, ncols=ncols, sharey=sharey,
        figsize=(x_size, y_size), gridspec_kw={'wspace':wspace})
    else:
        fig, ax = plt.subplots(figsize=(x_size, y_size))
    
    return fig, ax


# In[4]:


def extract_ttype_df(df, epoch_events, trialtype, all_trialtypes):
    ttype_list = []
    for t in all_trialtypes[trialtype]:
        ttype_list.append('Tactile '+t)
        ttype_list.append('Auditory '+t)
    ttype_df = df[df.index.get_level_values('Final Outcome').isin(ttype_list)]  
    ttype_events = epoch_events[epoch_events['Final Outcome'].isin(ttype_list)]
    
    return ttype_df, ttype_events


# In[5]:


def plot_behavioral_vlines(outcome, trialtype, response_latency, ax_obj, x_range, fps):
    if ('miss' not in outcome) & ('CR' not in outcome):
        response_frame = float(response_latency)/fps

        if ('reward' in trialtype) | ('omission' in trialtype) | ('outcomes' in trialtype):
            ax_obj.axvline(response_frame+fps, color='darkcyan', alpha=0.75, linewidth=1)
        else:
            ax_obj.axvline(response_frame, color='darkorange', alpha=0.75, linewidth=1)                                    
    else:
        response_frame = 3.5*fps
        ax_obj.axvline(response_frame, color='darkorange', alpha=0.75, linewidth=1)   
    ax_obj.axvline(3*fps, color='indianred', alpha=0.75, linewidth=1) if x_range[0] <= 3 else None 


# In[6]:


def sort_deltaF_rows(trial, channel, trial_df, all_events, ttype_events, sort_by=['nEvents','peak'], ascending=[False,False], events_first=False):
    ## Find all events for current trial and integrate into trial_df ##
    all_trial_events = all_events[all_events['Unique_trial'] == trial]
    ROI_event_counts = all_trial_events.groupby(all_trial_events['Unique_ROI']).size()
    ROI_event_counts = ROI_event_counts[ROI_event_counts.index.str.contains(channel)].to_frame()
    ROI_event_counts.index.names = ['Unique_ROI']
    ROI_event_counts.columns = ['nEvents']
    ROI_idx_df = trial_df.reset_index().set_index('Unique_ROI')
    event_integrated_trial_df = pd.concat([ROI_idx_df, ROI_event_counts], axis=1)
    event_integrated_trial_df['nEvents'] = event_integrated_trial_df['nEvents'].fillna(0)
    idx_names = [x if x != None else 'level_0' for x in list(trial_df.index.names)]
    event_integrated_trial_df = event_integrated_trial_df.reset_index().set_index(idx_names)

    ## Find epoch events for current trial and make a separate deltaF dataframe (to plot last) ## ##                               
    epoch_trial_events = ttype_events[ttype_events['Unique_trial'] == trial]
    event_ROIs = epoch_trial_events['Unique_ROI'].unique()
    event_df = event_integrated_trial_df[event_integrated_trial_df.index.get_level_values('Unique_ROI').isin(event_ROIs)]
    non_event_df = event_integrated_trial_df[~event_integrated_trial_df.index.get_level_values('Unique_ROI').isin(event_ROIs)]

    ## Calculate peak, peak_time, skew and sort by metrics of choice ##   
    event_df['peak'] = event_df.max(axis=1)
    event_df['peak_time'] = event_df.idxmax(axis=1)
    event_df['skew'] = event_df.skew(axis=1)
    
    non_event_df['peak'] = non_event_df.max(axis=1)
    non_event_df['peak_time'] = non_event_df.idxmax(axis=1)
    non_event_df['skew'] = non_event_df.skew(axis=1)
    
    event_df = event_df.sort_values(sort_by, ascending=ascending).iloc[:,:-4] 
    non_event_df = non_event_df.sort_values(sort_by, ascending=ascending).iloc[:,:-4]    

    ## Concatenate event and non-event deltaF dataframes, and find corresponding indexes ##
    if events_first == False:
        final_data = pd.concat([non_event_df, event_df])
        non_event_indexes = list(range(non_event_df.shape[0]))
        event_indexes = list(range(non_event_df.shape[0], non_event_df.shape[0]+event_df.shape[0]))
    else:
        final_data = pd.concat([event_df, non_event_df])
        event_indexes = list(range(event_df.shape[0]))
        non_event_indexes = list(range(event_df.shape[0], event_df.shape[0]+non_event_df.shape[0]))        
    
    return final_data, non_event_indexes, event_indexes

