#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import time
import math
import matplotlib.backends.backend_pdf
import pickle
from collections import Counter
from scipy.stats import rankdata, mannwhitneyu, ttest_ind, ttest_rel, spearmanr, zscore, pearsonr, wilcoxon
from scipy.integrate import trapz
from random import sample
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# ## Small math functions

# In[2]:

def set_data_directory(directory):
    global data_directory
    data_directory = directory

    return directory

def upload_cohort_epochs(data_directory):
    global cohort_epochs
    cohort_epochs = {}
    ## Loop through every sub-directory in the data directory to find different datatypes (defined by the iterator "dtype") ##
    for dtype in ['events', 'deltaF', 'pupil']:
        cohort_dtype = {}
        for root, dirs, files in os.walk(data_directory):
            for f in files:
                if (dtype + '_epochs_V10.npy' in f) & ('Old version' not in root):
                    ID = root.split('\\')[-1]
                    # if (ID == 'RGECO_GCamP_Batch3_one') & ('PRE_MANUAL' not in f):
                    #     continue
                    # print(ID, f)
                    ID_dtype = np.load(root + '\\' + f, allow_pickle=True).item()

                    for epoch, epoch_df in ID_dtype.items():
                        if dtype not in cohort_epochs.keys():
                            cohort_epochs[dtype] = {}

                            if epoch not in cohort_epochs[dtype].keys():
                                cohort_epochs[dtype][epoch] = {}
                                cohort_epochs[dtype][epoch][ID] = epoch_df
                            else:
                                cohort_epochs[dtype][epoch][ID] = epoch_df
                        else:
                            if epoch not in cohort_epochs[dtype].keys():
                                cohort_epochs[dtype][epoch] = {}
                                cohort_epochs[dtype][epoch][ID] = epoch_df
                            else:
                                cohort_epochs[dtype][epoch][ID] = epoch_df

    return cohort_epochs

## Split time-series data "a" into "n" segments (i.e. downsampling to "n" samples)
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

## Calculate parametric Pearson correlation p-value ##
def calculate_pvalues(df, rtype):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4) if rtype == 'pearson' else round(spearmanr(tmp[r], tmp[c])[1], 4)
    return pvalues

## Calculate absolute distance between 2 values ##
def distance(x, y):
    if x >= y:
        result = x - y
    else:
        result = y - x
    return result

## Find most common-occurring element in list ##
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

## Calculate area under the curve of pandas dataframe time-series using the trapezoid rule ##
def calculate_area_under_curve(df):
    areas = []
    for _, row in df.iterrows():
        x = np.arange(len(row))
        y = row.values
        area = trapz(y, x)/900
        areas.append(area)
    return pd.Series(areas, index=df.index, name='Area').to_frame()


# ## Data manipulation functions

# In[3]:


def calculate_posteriors(group_df, size, protocol, dtype):
    if 'reversed outcomes' in list(group_df.index.names):
        outcome_label = 'reversed outcomes'
    else:
        outcome_label = 'Final Outcome'
        
    posteriors = {}
    for block in range(1000):        
        if block+size > group_df.shape[0]:
            break
        else:
            block_df = group_df[block:block+size]
    
            posteriors[block] = {}
            posteriors[block]['reward prob'] = {}
            posteriors[block]['stimulus ratios'] = {}
            posteriors[block]['response rate'] = {}
            posteriors[block]['rare prob'] = {}
            posteriors[block]['nresp'] = {}
            
            if 'det' not in protocol:
                Go_df = block_df[(block_df.index.get_level_values(outcome_label).str.contains('hit')) | (block_df.index.get_level_values(outcome_label).str.contains('miss'))]
                Go_ID_list = list(Go_df.index.get_level_values('Stimulus'))
                Go_ID = most_frequent(Go_ID_list)
                NoGo_ID = float(1) if Go_ID == float(2) else float(2)
                Go = block_df[block_df.index.get_level_values('Stimulus') == Go_ID]
                NoGo = block_df[block_df.index.get_level_values('Stimulus') == NoGo_ID]
            else:
                Go = block_df[block_df.index.get_level_values('Stimulus') != float(3)]
                NoGo = block_df[block_df.index.get_level_values('Stimulus') == float(3)]
                
            rew_probs = {}
            response_probs = {}
            responses = {}
            for m, modality in zip(['G','N'],[Go, NoGo]):
                nFA = modality[modality.index.get_level_values(outcome_label).str.contains('FA')].shape[0]
                nHIT = modality[modality.index.get_level_values(outcome_label).str.contains('hit')].shape[0]
                ntype = modality.shape[0]
                nrew = modality[modality.index.get_level_values('Water') == float(1)].shape[0]
                rew_prob = nrew/(nFA+nHIT) if nFA+nHIT > 2 else np.nan
                rew_probs[m] = rew_prob
                response_probs[m]  = (nFA+nHIT)/ntype
                responses[m] = nFA+nHIT
            rare = Go[(Go.index.get_level_values('Water') == float(0)) & (Go.index.get_level_values(outcome_label).str.contains('hit'))].shape[0] + NoGo[(NoGo.index.get_level_values('Water') == float(1)) & (NoGo.index.get_level_values(outcome_label).str.contains('FA'))].shape[0]
            frequent = Go[(Go.index.get_level_values('Water') == float(1)) & (Go.index.get_level_values(outcome_label).str.contains('hit'))].shape[0] + NoGo[(NoGo.index.get_level_values('Water') == float(0)) & (NoGo.index.get_level_values(outcome_label).str.contains('FA'))].shape[0]
            if (rare > 2) & (frequent > 2):
                posteriors[block]['rare prob'] = rare/frequent
            else:
                posteriors[block]['rare prob'] = np.nan
            posteriors[block]['reward prob'] = rew_probs
            posteriors[block]['response rate'] = response_probs
            posteriors[block]['nresp'] = responses
           
            ntrials = block_df.shape[0]
            posteriors[block]['stimulus ratios'] = {'G':Go.shape[0]/ntrials,
                                                    'N':NoGo.shape[0]/ntrials}
    
    return posteriors


# In[4]:


def reverse_outcomes(sesh_df):
    outcome_list = []
    for index, rows in sesh_df.iterrows():
        out = index[-1]
        if 'hit' in out:
            outcome_list.append(out.split(' hit')[0]+' FA')
        elif 'FA' in out:
            outcome_list.append(out.split(' FA')[0]+' hit')                     
        elif 'miss' in out:
            outcome_list.append(out.split(' miss')[0]+' CR')  
        elif 'CR' in out:
            outcome_list.append(out.split(' CR')[0]+' miss')
        else:
            print(out, 'unexpected outcome label')
    
    return outcome_list


# In[5]:


def recount_licks(df):
    prestim_licks = [len([x for x in licks if x < 3][::2]) for licks in list(df['Trial licks'])]
    ant_licks = [len([x for x in licks if latency+3 < x < latency+4][::2]) for latency, licks in zip(list(df['Response latency']), list(df['Trial licks']))]
    rew_licks = [len([x for x in licks if latency+4 < x < latency+5][::2]) for latency, licks in zip(list(df['Response latency']), list(df['Trial licks']))]
    delayed_rew_licks = [len([x for x in licks if latency+5 < x < latency+6][::2]) for latency, licks in zip(list(df['Response latency']), list(df['Trial licks']))]
    ITI_licks = [len([x for x in licks if ITI < x < ITI+1][::2]) for ITI, licks in zip(list(df['Timeout_end']), list(df['Post-trial licks']))]    
    recounted_df = pd.DataFrame([prestim_licks, ant_licks, rew_licks, delayed_rew_licks, ITI_licks], 
                                index=['Prestim licks','Anticipatory licks','Reward licks','Delayed_reward licks','ITI licks'],
                                columns=df.index).T
    return recounted_df


# In[6]:


def calculate_session_rates(posteriors_df, final_df, sesh, protocol, dtype, mouse):
    prop_rare = posteriors_df['reward prob']['N'] / posteriors_df['reward prob']['G']
    hit_rate = posteriors_df['response rate']['G']
    FA_rate = posteriors_df['response rate']['N']
    correct = final_df[(final_df['Final Outcome'].str.contains('hit')) | (final_df['Final Outcome'].str.contains('CR'))].shape[0]
    CP_rate = correct/final_df.shape[0]
    
    if len([x for x in list(final_df['Final Outcome']) if 'FA' in str(x)]) >= 3:
        FA = final_df[final_df['Final Outcome'].str.contains('FA')]
        FA_latency = (FA['Response latency']).mean()
        FA_licks = (FA['Anticipatory licks']).mean()
        FA_licks_rew = (FA[FA.Water == float(1)]['Reward licks']).mean() if FA[FA.Water == float(1)].shape[0] > 0 else np.nan
        FA_licks_om = (FA[FA.Water == float(0)]['Reward licks']).mean() if FA[FA.Water == float(0)].shape[0] > 0 else np.nan
        FA_licks_ITI = (FA['ITI licks']).mean()
    else:
        FA_latency = np.nan
        FA_licks = np.nan
        FA_licks_rew = np.nan
        FA_licks_om = np.nan
        FA_licks_ITI = np.nan

    if len([x for x in list(final_df['Final Outcome']) if 'hit' in str(x)]) >= 3:
        hit = final_df[final_df['Final Outcome'].str.contains('hit')]
        hit_latency = (hit['Response latency']).mean()
        hit_licks = (hit['Anticipatory licks']).mean()
        hit_licks_rew = (hit[hit.Water == float(1)]['Reward licks']).mean() if hit[hit.Water == float(1)].shape[0] > 0 else np.nan    
        hit_licks_om = (hit[hit.Water == float(0)]['Reward licks']).mean() if hit[hit.Water == float(0)].shape[0] > 0 else np.nan    
        hit_licks_ITI = (hit['ITI licks']).mean()      
    else:
        hit_latency = np.nan
        hit_licks = np.nan
        hit_licks_rew = np.nan
        hit_licks_om = np.nan
        hit_licks_ITI = np.nan
        
    all_rew = (final_df[final_df.Water == float(1)]['Reward licks']).mean() if final_df[final_df.Water == float(1)].shape[0] > 0 else np.nan
    all_om = (final_df[final_df.Water == float(0)]['Reward licks']).mean() if final_df[final_df.Water == float(0)].shape[0] > 0 else np.nan
    norm_licks = hit_licks/FA_licks if FA_licks > 0 else np.nan
    final_df = pd.DataFrame([CP_rate, hit_rate, FA_rate, prop_rare, norm_licks, hit_licks, FA_licks, all_rew, all_om, hit_licks_rew, FA_licks_rew, hit_licks_om, FA_licks_om, hit_licks_ITI, FA_licks_ITI, hit_latency, FA_latency, protocol], 
                            index=['CP rate','Hit rate','FA rate','Rare outcome proportion','Hit/FA licks', 'Hit ant', 'FA ant','All rew','All om', 'Hit rew','FA rew', 'Hit om', 'FA om','Hit ITI','FA ITI', 'Hit latency','FA latency', 'Protocol'], 
                            columns=[mouse+'_'+sesh]).T
    final_df = final_df.reset_index().set_index(['index','Protocol'])

    return final_df


# In[7]:


def reverse_outcome_indexes(df, dtype, reversed_sessions):
    if dtype == 'events':
        reversed_df = df[(df.index.get_level_values(0).isin(reversed_sessions.keys())) & (df['Unique session'].isin(list(reversed_sessions.values())[0]))]
        unreversed_df = df[~((df.index.get_level_values(0).isin(reversed_sessions.keys())) & (df['Unique session'].isin(list(reversed_sessions.values())[0])))]
    else:
        reversed_df = df[(df.index.get_level_values(0).isin(reversed_sessions.keys())) & (df.index.get_level_values('Unique session').isin(list(reversed_sessions.values())[0]))]
        unreversed_df = df[~((df.index.get_level_values(0).isin(reversed_sessions.keys())) & (df.index.get_level_values('Unique session').isin(list(reversed_sessions.values())[0])))]
    n_cols = reversed_df.reset_index().shape[1]-reversed_df.shape[1]
    reversed_df_drop = reversed_df.reset_index()
    reversed_df_drop['Final Outcome'] = reversed_df_drop['Final Outcome'].replace({'Tactile rewarded hit':'Tactile rewarded FA',
                                       'Tactile rewarded FA':'Tactile rewarded hit',
                                       'Auditory rewarded hit':'Auditory rewarded FA',
                                       'Auditory rewarded FA':'Auditory rewarded hit',
                                       'Tactile unrewarded hit':'Tactile unrewarded FA',
                                       'Tactile unrewarded FA':'Tactile unrewarded hit',
                                       'Auditory unrewarded hit':'Auditory unrewarded FA',
                                       'Auditory unrewarded FA':'Auditory unrewarded hit',
                                       'Tactile unrewarded miss':'Tactile unrewarded CR',
                                       'Tactile unrewarded CR':'Tactile unrewarded miss',
                                       'Auditory unrewarded miss':'Auditory unrewarded CR',
                                       'Auditory unrewarded CR':'Auditory unrewarded miss'})
    reversed_df = reversed_df_drop.set_index(list(reversed_df_drop.columns)[:n_cols])
    fixed_df = pd.concat([reversed_df, unreversed_df])
    fixed_df.index.names = df.index.names    
    
    return fixed_df


# In[8]:


def extract_reward_ITI(index, lick_ind, water_ind, ITI_ind):
    if index[lick_ind] == '-':
        water = 4.5
        ITI = 8
    else:
        if index[water_ind] == '-':
            water = int(index[lick_ind])/1000 + 1
        else:
            water = int(index[water_ind])/1000
        if index[ITI_ind] == '-':
            ITI = 8
        else:
            ITI = int(index[ITI_ind])/1000
            
    return water, ITI


# In[9]:


def centre_event_latency(df, epoch):
    if epoch == 'trial_onset':
        df['ev_latency'] = df['ev_onset']
    elif epoch == 'prestim':
        df['ev_latency'] = df['ev_onset'] - 3
    elif epoch == 'stimulus':
        df['ev_latency'] = df['ev_onset'] - 3
    elif epoch == 'response':
        df['ev_latency'] = df['ev_onset'] - df['Response latency']/1000
    elif 'reward' in epoch:
        df['ev_latency'] = df['ev_onset'] - df['Water_time']/1000
    elif epoch == 'ITI':
        df['ev_latency'] = df['ev_onset'] - df['Timeout_end']/1000
    else:
        df['ev_latency'] = df['ev_onset']

    return df


# In[10]:


def get_UOA_event_metrics(df, label, cohort_epochs, ttype_list, ttype, min_trials):
    UOA_groups = df.groupby([df.index.get_level_values(label), df.index.get_level_values('level_0')])
    event_metrics = UOA_groups.mean()
    event_counts = UOA_groups.size().to_frame()
    all_deltaF = pd.concat(cohort_epochs['deltaF']['all'])
    ttype_deltaF = all_deltaF[(all_deltaF.index.get_level_values('Final Outcome') == 'Tactile '+ttype_list[0]) | 
                              (all_deltaF.index.get_level_values('Final Outcome') == 'Auditory '+ttype_list[0]) |
                              (all_deltaF.index.get_level_values('Final Outcome') == 'Tactile '+ttype_list[1]) | 
                              (all_deltaF.index.get_level_values('Final Outcome') == 'Auditory '+ttype_list[1])]
#     mutual_deltaF = ttype_deltaF[ttype_deltaF.index.get_level_values(label).isin(event_counts.index.get_level_values(label).unique())]
    exclusive_deltaF = ttype_deltaF[~ttype_deltaF.index.get_level_values(label).isin(event_counts.index.get_level_values(label).unique())]
    exclusive_ROIs = list((exclusive_deltaF.index.get_level_values(0)+'-'+exclusive_deltaF.index.get_level_values(label)).unique())
    non_active_events = pd.DataFrame([0]*len(exclusive_ROIs), columns=[0], index=exclusive_ROIs)
    idx = pd.MultiIndex.from_tuples([tuple(x.split('-')) for x in list(non_active_events.index)])
    non_active_events.index = idx
    non_active_events.index.names = ['level_0', 'Unique_ROI']
    non_active_events = non_active_events.swaplevel(0,1)
    all_ROI_events = pd.concat([non_active_events, event_counts])
    all_ROI_events = all_ROI_events.rename({0:'nEvents'}, axis=1)
    trial_counts = ttype_deltaF.groupby([ttype_deltaF.index.get_level_values(label), ttype_deltaF.index.get_level_values(0)]).size().to_frame()
    trial_counts = trial_counts.rename({0:'nTrials'}, axis=1)
    rate_df = pd.concat([all_ROI_events, trial_counts], axis=1)
    rate_df = rate_df[rate_df.nTrials >= min_trials]
    rate_df['rate'] = rate_df.nEvents/rate_df.nTrials
    event_metrics = pd.concat([rate_df, event_metrics], axis=1)
    event_metrics.index.names = ['Unique_ROI','level_0']
    
    return event_metrics


# In[11]:


def extract_UOA_metrics(epoch_df, dtype, ttypes, min_trials, thresh, epoch, correlated_ROI_list, modulated_ROIs):
    
    """Extract calcium/pupil metrics averaged across unit of analysis (UOA) for epochs/trial types of interest for every protocol separately
    
    Parameters:
        epoch_df (pandas): MultiIndex dataframe for epoch of interest (extracted from "cohort_epochs" dictionary)
        dtype (str): Datatype label (i.e. 'pupil' or 'events')
        ttypes (list): List of trial types that you want to calculate metrics for
        min_trials (int): Minimum number of trials for each trial type required for metric calculation
        thresh (float): pupil threshold to separate into constricted and dilated trials
    """

    if dtype == 'pupil':
        metric_dfs = {}
        ## Extract pupil metrics for single trials and single sessions as the unit of analysis ##
        for UOA_label, UOA_name in zip(['Unique_trial','Unique session'], ['single_trials','session_by_session']):
            pupil_dfs = {}
            for ttype, ttype_list in ttypes.items():
                ttype_df = epoch_df[(epoch_df.index.get_level_values('Final Outcome') == 'Tactile '+ttype_list[0]) | 
                                    (epoch_df.index.get_level_values('Final Outcome') == 'Auditory '+ttype_list[0]) |
                                    (epoch_df.index.get_level_values('Final Outcome') == 'Tactile '+ttype_list[1]) |
                                    (epoch_df.index.get_level_values('Final Outcome') == 'Auditory '+ttype_list[1])]
#                 ttype_df = std_filter(ttype_df, thresh)['positive']
                AUC = calculate_area_under_curve(ttype_df) #area under curve for each trial
                peak = ttype_df.max(axis=1).to_frame() #peak pupil dilation for each trial
                pos_AUC = calculate_area_under_curve(std_filter(ttype_df, thresh)['positive'])
                neg_AUC = calculate_area_under_curve(std_filter(ttype_df, thresh)['negative'])
                pos_peak = std_filter(ttype_df, thresh)['positive'].max(axis=1).to_frame()
                neg_peak = std_filter(ttype_df, thresh)['negative'].min(axis=1).to_frame()           
                ttype_df = pd.concat([AUC, peak, pos_AUC, neg_AUC, pos_peak, neg_peak],axis=1)
                ttype_df.columns = ['AUC','peak','pos_AUC','neg_AUC','pos_peak','neg_peak']
                UOA_groups = ttype_df.groupby(ttype_df.index.get_level_values(UOA_label)) #split dataframe by UOA
                pupil_dfs[ttype] = UOA_groups.mean() #calculate mean for each of UOA (only relevant for session-by-session) and add to dictionary by trial type
            pupil_df = pd.concat(pupil_dfs, axis=1) #concatenate different trial type pandas together
            pupil_df['Protocol'] = [x.split('_')[0] for x in list(pupil_df.index)] #add protocol names as data column
            pupil_df = pupil_df.reset_index().set_index([UOA_label,'Protocol']).swaplevel(0,1,axis=1) #elevate UOA_names and protocol names to index column
            metric_dfs[UOA_name] = pupil_df #add metrics to dictionary by UOA type
            pupil_df.to_csv(data_directory+'\\'+('_').join([epoch,'peak_pupil',UOA_name,'.csv'])) #save final dataframe to data directory as csv
    elif dtype == 'events':
        metric_dfs = {}
        ## Extract calcium event metrics for ROIs as the unit of analysis ##
        UOA_label = 'Unique_ROI'
        events_dfs = {}
        epoch_df = epoch_df.reset_index().set_index(['level_0', UOA_label])
        epoch_df['Timeout_end'] = epoch_df['Timeout_end'].replace({'-':8000}) #cleanup timeout data column
        epoch_df = epoch_df[['Response latency','Water_time','Timeout_end','Final Outcome','peak','ev_onset','integral']] #extract data of interest from dataframe
        for ttype, ttype_list in ttypes.items():
            ttype_df = epoch_df[(epoch_df['Final Outcome'] == 'Tactile '+ttype_list[0]) | 
                                (epoch_df['Final Outcome'] == 'Auditory '+ttype_list[0]) |
                                (epoch_df['Final Outcome'] == 'Tactile '+ttype_list[1]) | 
                                (epoch_df['Final Outcome'] == 'Auditory '+ttype_list[1])]
            ttype_df = ttype_df.drop('Final Outcome',axis=1).astype(float) #only want numerical data remainining, then convert datatype to float
            centred_df = centre_event_latency(ttype_df, epoch) #centre event latency with respect to epoch of interest
            event_metrics = get_UOA_event_metrics(centred_df, UOA_label, cohort_epochs, ttype_list, ttype, min_trials) #extract all event metrics
            events_dfs[ttype] = event_metrics[['ev_latency','peak','integral','rate', 'nTrials']] #extract event metrics of interest and add to dictionary by trial type

        ## Concatenate different trial type pandas together, add new indexes, and save to data directory as csv ##
        event_df = pd.concat(events_dfs, axis=1)
        event_df = event_df.swaplevel(0,1,axis=1).sort_index(level=0, axis=1)
        event_df['Protocol'] = [x.split('_')[0] for x in list(event_df.index.get_level_values(UOA_label))]
        event_df['Channel'] = [x.split('_')[-2] for x in list(event_df.index.get_level_values(UOA_label))]
        event_df['ID'] = event_df.index.get_level_values('level_0')
        event_df['Unique_session'] = event_df['ID']+'_'+[('_').join(x.split('_')[:-2]) for x in list(event_df.index.get_level_values('Unique_ROI'))]
        event_df = event_df.reset_index().set_index(['ID',UOA_label,'Protocol', 'Channel', 'Unique_session']).drop('level_0',axis=1)
        correlated_ROI_df = event_df[event_df.index.get_level_values('Unique_ROI').isin(correlated_ROI_list)]
        uncorrelated_ROI_df = event_df[(~event_df.index.get_level_values('Unique_ROI').isin(correlated_ROI_list)) & (event_df.index.get_level_values('Unique_session').isin(correlated_ROI_df.index.get_level_values('Unique_session').unique()))]
        
        for df, label in zip([event_df, correlated_ROI_df, uncorrelated_ROI_df],['ALL_ROIs','correlated_ROIs','uncorrelated_ROIs']):
            metric_dfs[label] = df
            df.to_csv(data_directory+'\\'+('_').join([label, epoch,'ROI_by_ROI_calcium_metrics.csv']))    
            df.replace({0:np.nan}).to_csv(data_directory+'\\'+('_').join([label, epoch,'NON_ZERO_ROI_by_ROI_calcium_metrics.csv'])) #only keep active ROIs  
#             df_scaled = df.groupby(level=0, axis=1).transform(lambda x : (x-x.min())/(x.std())) #Z-score ROI-by-ROI, across trial types
#             df_scaled.to_csv(data_directory+'\\'+('_').join([label, epoch,'ZSCORED_ROI_by_ROI_calcium_metrics.csv']))                        
    return metric_dfs

def generate_all_metrics_file(min_trials, pupil_thresh, ttypes, reversed_sessions, correlated_ROI_list):
    all_metrics = {}
    for dtype, dtype_dict in cohort_epochs.items():
        if dtype != 'deltaF': #only want to get metrics of events and pupil
            all_metrics[dtype] = {}
            for epoch, epoch_dict in dtype_dict.items():
                epoch_df = pd.concat(epoch_dict)
                if epoch_df.shape[0] > 0: #if there are more than 0 trials in this dataframe
                    if (epoch == 'all') & (dtype == 'pupil'):
                        epoch_df = epoch_df.loc[:,(5*900):(7*900)]
                    epoch_df = reverse_outcome_indexes(epoch_df, dtype, reversed_sessions)
                    metric_dfs = extract_UOA_metrics(epoch_df, dtype, ttypes, min_trials, pupil_thresh, epoch, correlated_ROI_list, modulated_ROIs) #save UOA metrics to dictionary "metric_dfs", and also as csv to data directory
                    all_metrics[dtype][epoch] = metric_dfs #add "metric_dfs" to final dictionary containing all metrics from all epochs of all datatypes "all_metrics"
    return all_metrics

# ## Plotting functions

# In[12]:


def plot_binned_licks(df, trialtypes, ttype_colors, nrows, ncols, positions, protocols_to_plot, binsize, savename=None):
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(6.44+0.1,4.48), sharey=True, gridspec_kw={'wspace':0.01}, constrained_layout=True)
    for protocol in protocols_to_plot:
        tdf = df[df.Protocol.str.contains(protocol)]
        binned_trial = pd.concat([pd.DataFrame(np.histogram(x[::2], np.linspace(0,8,int(8/binsize)+1))[0]) for x in tdf['Trial licks'].to_numpy()], axis=1).T
        binned_post_trial = pd.concat([pd.DataFrame(np.histogram(x[1:][::2], np.linspace(8.065,(12.065+(1*binsize)),int(5/binsize)+1))[0]) for x in tdf['Post-trial licks'].to_numpy()], axis=1).T
        binned_licks = pd.concat([binned_trial, binned_post_trial], axis=1).T.reset_index().T.iloc[1:]
        binned_licks.index = pd.MultiIndex.from_tuples([(x,y) for x,y in zip(list(tdf['Final Outcome']), list(tdf['Unique session']))])
        binned_licks = binned_licks/binsize
        binned_licks.columns = [x+(binsize/2) for x in list(binned_licks.columns)]

        p = positions[protocol]
        for t,ttypes in trialtypes.items():
            out_df = binned_licks[(binned_licks.index.get_level_values(0) == 'Tactile '+ttypes[0]) |
                                  (binned_licks.index.get_level_values(0) == 'Auditory '+ttypes[0]) |
                                  (binned_licks.index.get_level_values(0) == 'Tactile '+ttypes[1]) |
                                  (binned_licks.index.get_level_values(0) == 'Auditory '+ttypes[1])].droplevel(0)
            out_df = out_df.groupby(out_df.index).mean()
            ax[p[0],p[1]].plot(out_df.mean(), color=ttype_colors[t], linewidth=2)

        for t,ttypes in trialtypes.items():
            out_df = binned_licks[(binned_licks.index.get_level_values(0) == 'Tactile '+ttypes[0]) |
                                  (binned_licks.index.get_level_values(0) == 'Auditory '+ttypes[0]) |
                                  (binned_licks.index.get_level_values(0) == 'Tactile '+ttypes[1]) |
                                  (binned_licks.index.get_level_values(0) == 'Auditory '+ttypes[1])].droplevel(0)
            out_df = out_df.groupby(out_df.index).mean()
            ax[p[0],p[1]].fill_between(out_df.columns, out_df.mean()-out_df.sem(), out_df.mean()+out_df.sem(), alpha=0.15, color=ttype_colors[t])
        ax[p[0],p[1]].set_xlim(0,int(12/binsize))
        ax[p[0],p[1]].set_xticks(np.linspace(0,int(12/binsize),13))
        ax[p[0],p[1]].set_xticklabels([str(int(x*binsize)) for x in np.linspace(0,int(12/binsize),13)], fontsize=11)
        ax[p[0],p[1]].set_title(protocol)
        ax[p[0],0].set_ylabel('Lick rate (Hz)', fontsize=11)
    if savename == None:
        None
    else:        
        fig.savefig(data_directory+'\\'+savename+'.svg',  bbox_inches='tight')


# In[13]:


def plot_lick_rasters(df, raster_types, raster_colors, specific_session=None):
#     for protocol in df['Protocol'].unique():
#         tdf = df[df['Protocol'].str.contains(protocol)]
        
    if specific_session == None:
        unique_sessions = df.groupby([df['Unique session'], df['ID']]).mean()
        session_list = [{'ID':x, 'session':y} for x, y in zip(list(unique_sessions.index.get_level_values('ID')),list(unique_sessions.index.get_level_values('Unique session')))]
    else:
        session_list = specific_session

    for sesh in session_list:
        ID = sesh['ID']
        session = sesh['session']
        session_df = df[df['Unique session'] == session]
        protocol = session.split('_')[0]

        ## First extract session-averaged lick metrics (not necessarily only for the trialtypes to be plotted, but all trialtypes) ##
        response_times = [3.5 if x == '-' else float(x)+3 for x in list(session_df['Response latency'])]
        ITI_times = [8 if x == '-' else float(x) for x in list(session_df['Timeout_end'])]
        variable_times = pd.DataFrame([response_times, [x+1 for x in response_times], ITI_times], index=['Response latency', 'Water_time','Timeout_end']).T
        variable_times['Stimulus'] = [3]*variable_times.shape[0]

        fig = plt.figure()
        row = 0
        for out, out_list in raster_types.items():
            color = raster_colors[out]
            out_df = session_df[(session_df['Final Outcome'] == 'Tactile '+out_list[0]) |
                                (session_df['Final Outcome'] == 'Auditory '+out_list[0]) |
                                (session_df['Final Outcome'] == 'Tactile '+out_list[1]) |
                                (session_df['Final Outcome'] == 'Auditory '+out_list[1])]
            for index, rows in out_df.iterrows():
                tl = rows['Trial licks'][::2]
                pl = rows['Post-trial licks'][::2]
                plt.vlines(tl, [row]*len(tl), [row+1]*len(tl), color=color)
                plt.vlines(pl, [row]*len(pl), [row+1]*len(pl), color=color)
                row += 1

        for name, color in zip(['Stimulus', 'Water_time', 'Timeout_end'], ['red','green','black']):            
            data = variable_times[name]

            if name == 'Stimulus':
                ls = '-'
            elif name == 'Timeout_end':
                if 'unreward' in out:
                    ls = '--'
                else:
                    ls = '-'
            else:
                ls = '--'

            plt.axvline(data.mean(), color=color, linestyle=ls, alpha=0.5)
            plt.errorbar(data.mean(), row+1, xerr=data.std(), color=color, fmt='o', markersize=3) if ls == '--' else None
        plt.suptitle(ID+'_'+session)
        
        if specific_session == None:
            None
        else:
            fig.savefig(data_directory+'\\'+ID+'_'+session+'_lick_raster.svg')


# In[14]:


def plot_and_extract_correlated_ROIs(cohort_deltaF, cohort_events, r_thresh, event_thresh, time_thresh):
    
    """Plots correlated ROI pairs from "cohort_correlation" dict (individual mouse .npy files saved elsewhere in a previous script), and saves plots as a pdf. 
    There is also the option of re-defining new correlation criteria, and saves these correlation metrics as a .npy file.
    
    Parameters
        r_thresh (float): minimum Pearson r correlation coeffcient required for a pair of ROIs to be considered correlated
        event_thresh (float): minimum proportion of shared calcium events required for a pair of ROIs to be considered correlated
        time_thresh (int): maximum number of frames between calciume events to be considered a "shared events"
        
    Returns:
        Plots correlated ROIs session traces if the pdf for the given correlation criteria doesn't already exist.
    """
    ## Upload cohort files ##
    cohort_correlations = np.load(data_directory+'\\cohort_correlations.npy', allow_pickle=True).item()
    cohort_params = np.load(data_directory+'\\cohort_params.npy', allow_pickle=True).item()

    ## Pdf file naming/directory setup ##
    fname = 'correlated_ROIs_'+str(r_thresh)+'rthresh_'+str(time_thresh)+'timethresh_'+str(event_thresh)+'eventthresh' #pdf filename based on correlation criteria
    
    if fname+'.npy' in os.listdir(data_directory):
        print(fname+'.npy already exists') #no function performed if pdf already exists for the current criteria
    else:
        # pdf = matplotlib.backends.backend_pdf.PdfPages(data_directory+'\\'+fname+'.pdf') #open up a new pdf with the new filename
        
        ## Loop through nested structure of input dictionaries (mouse --> session --> ROI) ##
        correlated_ROIs = {} #final dictionary to add correlation metrics to (with new correlation criteria considered)
        for mouse, mouse_dict in cohort_correlations.items():
            correlated_ROIs[mouse] = {}
            if 'cross-pairs' in mouse_dict.keys(): #look specifically at correlated pairs of ROIs across channels (as determined by previous script)
                for session, session_dict in mouse_dict['cross-pairs'].items():
                    session_split = session.split('_') #session name should be in the format "protocol_date_FOV"
                    FPS = cohort_params[mouse][session_split[0]][session_split[1]][session_split[2]]['FPS'] #get image acquisition framerate from "cohort_params" dict
                    
                    ## For each ROI, extract the deltaF/events ##
                    for ROI, ROI_dict in session_dict.items():
                        mouse_df = cohort_deltaF[mouse] #mouse deltaF (i.e. all ROIs/sessions) of the current ROI
                        ROI_df = mouse_df[mouse_df.index.get_level_values('Unique_ROI') == ROI] #deltaF of the current ROI
                        trials = ROI_df.index.get_level_values('Trial') #trial numbers of the current ROI
                        ROI_df = (ROI_df - ROI_df.min().min()) / (ROI_df.max().max() - ROI_df.min().min()) #normalized deltaF of each trial for the current ROI
                        ROI_flat = ROI_df.to_numpy().flatten() #concatenate all trials of deltaF and flatten to 1-d array
                        mouse_events = cohort_events[mouse] #mouse events (i.e. all ROIs/sessions) of the current ROI
                        ROI_events = mouse_events[mouse_events['Unique_ROI'] == ROI] #events of the current ROI
                        
                        ## Find peak times of all events for current ROI ##
                        ROI_peak_times = []
                        for index, rows in ROI_events.iterrows():
                            trial_idx = np.where(np.asarray(trials) == rows.Trial)[0][0] #find deltaF index of the trial corresponding to the current event in the loop
                            peak_time = (trial_idx*ROI_df.shape[1]) + (rows['peak_time']*FPS) #(trial number*nframes per trial) + (peak time for current trial*framerate) to determine peak frame number for session trace
                            ROI_peak_times.append(peak_time) if peak_time < ROI_flat.shape[0] else None 
                        ROI_peaks = [ROI_flat[math.ceil(x)] for x in ROI_peak_times] #use peak times to get peak deltaF values from "ROI_flat"

                        ROI_color = 'green' if 'Green' in ROI else 'red'
                        ROI_label = ('_').join(ROI.split('_')[-2:])
                        
                        ## Extract deltaF/events and find peak times of all events for other ROIs in the same FOV (i.e. partner ROIs) ##
                        for partner, r in ROI_dict.items(): #ROI_dict contains every ROI (i.e. partner) that was correlated with the current ROI (from previous script)
                            if r > r_thresh: #Pearson r threshold
                                partner_df = mouse_df[mouse_df.index.get_level_values('Unique_ROI') == partner]
                                partner_df = (partner_df - partner_df.min().min()) / (partner_df.max().max() - partner_df.min().min())
                                partner_flat = partner_df.to_numpy().flatten()
                                partner_flat = partner_flat-0.75 #subtract 0.75 from normalized deltaF of partner ROI for plotting purposes (i.e. to view stacked on top of each other)

                                partner_events = mouse_events[mouse_events['Unique_ROI'] == partner]
                                partner_peak_times = []
                                for index, rows in partner_events.iterrows():
                                    trial_idx = np.where(np.asarray(trials) == rows.Trial)[0][0]
                                    peak_time = (trial_idx*partner_df.shape[1]) + (rows['peak_time']*FPS)
                                    partner_peak_times.append(peak_time) if peak_time < partner_flat.shape[0] else None
                                partner_peaks = [partner_flat[math.ceil(x)] for x in partner_peak_times]

                                partner_color = 'green' if 'Green' in partner else 'red'
                                partner_label = ('_').join(partner.split('_')[-2:])
                                
                                ## Find shared events between current ROI and current partner ROI ##
                                matched_event_times = []
                                matched_event_peaks = []
                                matched_event_diffs = []
                                for ROI_time, ROI_peak in zip(ROI_peak_times, ROI_peaks):
                                    event_match_diff = []
                                    event_match_times = []
                                    event_match_peaks = []
                                    for partner_time in partner_peak_times:
                                        if distance(ROI_time, partner_time) <= time_thresh: #if difference in event peak time is <= the max threshold
                                            if len(event_match_diff) == 0:
                                                event_match_diff.append(distance(ROI_time, partner_time))
                                                event_match_times.append(ROI_time)
                                                event_match_peaks.append(ROI_peak+0.1) #want to plot an asterisk 0.1 units above the actual peak of the trace
                                            else:
                                                if distance(ROI_time, partner_time) < event_match_diff[0]:
                                                    event_match_diff.append(distance(ROI_time, partner_time))
                                                    event_match_times.append(ROI_time)
                                                    event_match_peaks.append(ROI_peak + 0.1)
                                                else:
                                                    None
                                    matched_event_times.append(event_match_times[0]) if len(event_match_times) > 0 else []
                                    matched_event_peaks.append(event_match_peaks[0]) if len(event_match_peaks) > 0 else []
                                    matched_event_diffs.append(event_match_diff[0]) if len(event_match_diff) > 0 else []

                                ## Calculate shared event ratio, and plot ROI-pair if it is above the threshold ##
                                event_ratio = len(matched_event_times) / len(ROI_peak_times)
                                event_diffs = sum(matched_event_diffs) / len(matched_event_diffs) if len(matched_event_diffs) > 0 else np.nan
                                if event_ratio > event_thresh:
                                    
                                    ## Store Pearson r and ratio of shared events in dictionary ##
                                    if ROI not in correlated_ROIs[mouse].keys():
                                        correlated_ROIs[mouse][ROI] = {}
                                        correlated_ROIs[mouse][ROI][partner] = {'r':r, 'event_ratio':event_ratio, 'mean_diff':event_diffs}
                                    else:
                                        correlated_ROIs[mouse][ROI][partner] = {'r':r, 'event_ratio':event_ratio, 'mean_diff':event_diffs}
                                    
                                    # ## Plotting parameters ##
                                    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20,4))
                                    # fig.suptitle(mouse+'_'+session+' \n '+ROI_label, fontsize=25, y=0.98)
                                    # ax.plot(partner_flat, color=partner_color, alpha=0.3)
                                    # ax.plot(ROI_flat, color=ROI_color,alpha=0.3)
                                    #
                                    # ax.scatter(y=ROI_peaks, x=ROI_peak_times, color=ROI_color, marker='*', s=15)
                                    # ax.scatter(y=partner_peaks, x=partner_peak_times, color=partner_color, marker='*', s=15)
                                    # ax.scatter(y=matched_event_peaks, x=matched_event_times, color='orange', marker='*', s=25)
                                    #
                                    # ax.set_title(partner_label, fontsize=15)
                                    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        np.save(data_directory+'\\'+fname+'.npy', correlated_ROIs)
        # pdf.close()


def filter_correlated_ROIs(correlated_ROIs, pairwise_cutoff, cutoff_on, nbins, display_plot):
    ## Loop through all ROI-pairs in correlated_ROIs dict and construct pairwise-event ratio dataframe ##
    correlation_event_ratios = {}
    for ID, ID_dict in correlated_ROIs.items():
        ID_dfs = []
        for ROI, ROI_dict in ID_dict.items():
            ROI_ratios = []
            ROI_pair_names = []
            for pair, pair_dict in ROI_dict.items():
                ROI_ratios.append(pair_dict['event_ratio'])
                ROI_pair_names.append(pair)
            ROI_df = pd.DataFrame(ROI_ratios, index=ROI_pair_names, columns=[ROI])
            ID_dfs.append(ROI_df)
        ID_df = pd.concat(ID_dfs, axis=1) if len(ID_dfs) > 0 else pd.DataFrame([])
        correlation_event_ratios[ID] = ID_df
    event_ratio_df = pd.concat(correlation_event_ratios)

    filtered_ROI_dfs = {}
    for channel in ['Red', 'Green']:
        ## First split the data into individual pairwise, ROI-by-ROI mean, and ROI-by-ROI max event ratios ##
        channel_ratio = event_ratio_df[event_ratio_df.index.get_level_values(1).str.contains(channel)]
        channel_means = channel_ratio.mean(axis=1)
        channel_max = channel_ratio.max(axis=1)

        ## Convert the 3 datatypes into 1-d numpy arrays ##
        channel_ratio_array = channel_ratio.to_numpy().flatten()
        channel_array = channel_ratio_array[~np.isnan(channel_ratio_array)]
        mean_array = channel_means.to_numpy()
        max_array = channel_max.to_numpy()

        ## Calculate the pairwise cutoff (i.e. quantile value of individual pairwise event ratios) ##
        cutoff = np.quantile(channel_array, pairwise_cutoff) if pairwise_cutoff != 0 else -0.1
        nROIs_individual = channel_array[channel_array > cutoff].shape[
            0]  # ROIs remaining if cutoff applied to individual pairwise event ratios
        nROIs_means = mean_array[mean_array > cutoff].shape[
            0]  # ROIs remaining if cutoff applied to ROI-by-ROI mean event ratios
        nROIs_maxes = max_array[max_array > cutoff].shape[
            0]  # ROIs remaining if cutoff applied to ROI-by-ROI max event ratios

        ## Calculate the min/max values across all 3 datatypes and calculate the bins to be used for all 3 histograms ##
        xmin = np.min(np.hstack([channel_array, mean_array, max_array]))
        xmax = np.max(np.hstack([channel_array, mean_array, max_array]))
        bins = np.linspace(xmin, xmax, nbins + 1)

        if display_plot == True:
            ## Plot the resulting histograms ##
            sns.set()
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 2), sharey=True, sharex=True)

            ax[0].hist(channel_array, bins=bins, density=True, alpha=0.7)
            ax[0].set_title('individual pairwise event ratios', fontsize=10)

            ax[1].hist(mean_array, bins=bins, density=True, alpha=0.7)
            ax[1].set_title('mean ROI-by-ROI event ratios', fontsize=10)

            ax[2].hist(max_array, bins=bins, density=True, alpha=0.7)
            ax[2].set_title('max ROI-by-ROI event ratios', fontsize=10)

            ## Loop to plot similar annotations across all subplots ##
            for col, ROIs in zip(range(3), [nROIs_individual, nROIs_means, nROIs_maxes]):
                ROI_label = 'ROI pairs' if col == 0 else 'ROIs'
                ax[col].axvline(cutoff)
                ax[col].text(x=ax[col].get_xlim()[1], y=ax[col].get_ylim()[1],
                             s=f'{ROIs} {ROI_label} > cutoff\n({pairwise_cutoff} quantile = {cutoff:.{3}f})', fontsize=8,
                             horizontalalignment='right', verticalalignment='top')
            plt.suptitle(channel, fontsize=12, y=1.1)
            plt.show()
        else:
            None

        ## Apply cutoff on desired dataset to return filtered ROI dataframe with corresponding event ratio metric (i.e. mean or max event ratio) ##
        if cutoff_on == 'maxes':
            filtered_ROIs = channel_max[channel_max > cutoff].to_frame()
        elif cutoff_on == 'means':
            filtered_ROIs = channel_means[channel_means > cutoff].to_frame()
        filtered_ROI_dfs[channel] = filtered_ROIs

    return filtered_ROI_dfs


def plot_ROI_by_ROI_traces(cohort_epochs, protocols, channels, trialtypes, colors, specific_ROIs, epoch, pre_window, post_window, plottype, reversed_sessions, figsize, save_ROIs=None):
    
    """Plots ROI-by-ROI traces (average overlaid on top of individual trials) for different trial types side-by-side for protocols/epochs of interest
    
    Parameters
        cohort_epochs (dict): input data from data uploading section
        protocols (list): list of protocols that you want to plot (separate plots)
        channels (list): list of channels that you want to plot (separate plots)
        trialtypes (list or dict) : list of trial types that you want to plot (side-by-side), or dictionary of lists where trialtypes within list will be aggregated
        colors (list) : list of corresponding plot colors for each trial type
        specific_ROIs (list or int) : list of ROI IDs that you want to plot specifically, or the n random ROIs
        epoch (str) : the particular epoch of the trial that you want the plots to be centred around (i.e. t=0)
        pre_window (int): number of frames before your epoch of interest that you want to plot
        post_window (int): number of frames after your epoch of interest that you want to plot
        
    Returns:
        Option to save traces as .svg file in the last line of this function (uncomment if you want that option)
    """
    
    df = pd.concat(cohort_epochs['deltaF']['all']) #easiest to plot if we use 'all' epoch instead of specific epochs which have restricted time window
    df = reverse_outcome_indexes(df, 'deltaF', reversed_sessions)
    
    ## Loop through each protocol of interest, channel of interest, then ROI of interest ##
    for protocol in protocols:
        protocol_df = df[df.index.get_level_values('Protocol') == protocol]
        for channel in channels:
            channel_df = protocol_df[protocol_df.index.get_level_values('Channel') == channel]
            ROI_groups = channel_df.groupby(channel_df.index.get_level_values('Unique_ROI'))
            
            ## Define the list of ROIs that we want to loop through (either specified in input parameters, or loop through every ROI) ##
            if type(specific_ROIs) is int:
                roi_list = sample(list(channel_df.index.get_level_values('Unique_ROI').unique()),specific_ROIs)
            elif (specific_ROIs == None) | (len(specific_ROIs) == 0):
                roi_list = channel_df.index.get_level_values('Unique_ROI').unique()
            else:
                roi_list = specific_ROIs

            for roi in roi_list:
                if roi in list(channel_df.index.get_level_values('Unique_ROI').unique()):
                    ROI_df = ROI_groups.get_group(roi)
                    xsize = 4*len(trialtypes) if 4*len(trialtypes) <= 20 else 20
                    fig, ax = plt.subplots(nrows=1, ncols=len(trialtypes), figsize=figsize, sharey=True, frameon=False, gridspec_kw={'wspace':0.05})
                    if type(trialtypes) == list:
                        trialtypes = dict.fromkeys(trialtypes)
                    else:
                        trialtypes = trialtypes
                    
                    variable_data = {}
                    for col, (ttype, ttype_list) in enumerate(trialtypes.items()):
                        variable_data[col] = {}

                        if ttype_list == None:
                            ttype_df = ROI_df[(ROI_df.index.get_level_values('Final Outcome') == 'Tactile '+ttype) | (ROI_df.index.get_level_values('Final Outcome') == 'Auditory '+ttype)]
                        else:
                            types = []
                            for t in ttype_list:
                                types.append('Tactile ' + t)
                                types.append('Auditory ' + t)
                            ttype_df = ROI_df[(ROI_df.index.get_level_values('Final Outcome').isin(types))]
                        no_response_dropped = np.asarray([float(x)/1000 for x in list(ttype_df.index.get_level_values('Response latency')) if x != '-'])
                        response_times = [3.5 if x == '-' else float(x)/1000 for x in list(ttype_df.index.get_level_values('Response latency'))]
                        ITI_times = [8 if x == '-' else float(x)/1000 for x in list(ttype_df.index.get_level_values('Timeout_end'))]
                        variable_times = pd.DataFrame([response_times, [x+1 for x in response_times], ITI_times], index=['Response latency', 'Water_time','Timeout_end']).T
                        variable_times['Stimulus'] = [3]*variable_times.shape[0]
                        
                        inferred = []
                        nsamples = ttype_df.shape[1]
                        fps = 30.3 if nsamples == 364 else 30.54
                        trial_duration = nsamples / fps

                        ## Loop through all trials of the current ROI, for the current protocol/trial type of interest ##
                        for index, rows in ttype_df.iterrows():
                            water, ITI = extract_reward_ITI(index, 8, 13, 14)

                            ## Determine trial-by-trial start time of epoch of interest ##
                            if epoch == 'trial_onset':
                                inferred.append(rows.iloc[math.floor(0*fps):math.floor(0*fps)+post_window].to_frame().reset_index().iloc[:,1:].T) #no pre-window
                            else:
                                if epoch == 'reward':
                                    onset = water
                                elif epoch == 'response':
                                    onset = water-1
                                elif epoch == 'stimulus':
                                    onset = 3
                                elif epoch == 'trial_onset':
                                    onset = 0
                                elif epoch == 'delayed_reward':
                                    onset = water+1
                                elif epoch == 'ITI':
                                    onset = ITI
                                inferred.append(rows.iloc[math.floor(onset*fps)-pre_window:math.floor(onset*fps)+post_window].to_frame().reset_index().iloc[:,1:].T) #add pre- and post-windows
                        if len(inferred) > 0 : #make sure that the trial type of interest actually exists for thie particular protocol/ROI
                            inferred_df = pd.concat(inferred)

                            ## Plotting parameters ##
                            ax[col].set_title(ttype)
                            ax[col].set_xlabel('Peri-'+epoch+' time (frames)')
                            if plottype == 'single':
                                for trial, rows in inferred_df.iterrows():
                                    if rows.mean() > float(-0.7): #don't plot weird data where fluorescence is stuck at -1 for the whole trial
                                        ax[col].plot(rows, color='black', linewidth=1, alpha=0.3)
                            elif plottype == 'shaded':
                                ax[col].fill_between(list(range(inferred_df.shape[1])), inferred_df.mean()-inferred_df.sem(), inferred_df.mean()+inferred_df.sem(), alpha=0.3, color=colors[ttype], linewidth=0)
                            ax[col].plot(inferred_df.mean(), color=colors[ttype], linewidth=2)

                            ## Plot mean +/- std points of lick time, reward time, and ITI time on top of graph ##
                            if (epoch == 'trial_onset') & (post_window == 243):
                                for name, color in zip(['Stimulus','Response latency', 'Water_time'], ['red','darkorange','green']):
                                    variable_data[col][name] = {}
                                    if name == 'Stimulus':
                                        ls = '-'
                                    elif name == 'Timeout_end':
                                        if 'unreward' in ttype:
                                            ls = '--'
                                        elif ('unrewarded hit' in ttype_list) | ('unrewarded FA' in ttype_list):
                                            ls = '--'
                                        else:
                                            ls = '-'
                                    else:
                                        ls = '--'
                                        
                                    data = (variable_times*fps)[name]
                                    ax[col].axvline(data.mean(), color=color, linestyle=ls, alpha=0.5)
                                    ax[col].errorbar(data.mean(), ROI_df.max().max(), xerr=data.std(), color=color, fmt='o', markersize=3) if ls == '--' else None
#                                     ax[col].fill_between([data.mean()-data.std(), data.mean()+data.std()], ROI_df.min().min(), ROI_df.max().max(), color=color, alpha=0.1)
                            else:
                                pre_ticks_seconds = list(range(-(math.floor(pre_window / fps)), 0))
                                post_ticks_seconds = list(range(0, math.floor(pre_window / fps) + 1))
                                xticks_seconds = pre_ticks_seconds + post_ticks_seconds
                                xtick_labels = [str(x) for x in (xticks_seconds)]

                                pre_tick_locations = [pre_window + (x * fps) for x in pre_ticks_seconds]
                                post_tick_locations = [pre_window + (x * fps) for x in post_ticks_seconds]
                                xtick_locations = pre_tick_locations + post_tick_locations
                                
                                ax[col].set_xticks(xtick_locations)
                                ax[col].set_xticklabels(xtick_labels)
                                if epoch == 'stimulus':
                                    ax[col].axvline(pre_window, color='red', alpha=0.5)
                                    if ('miss' not in ttype) & ('CR' not in ttype):
                                        mean_latency = no_response_dropped.mean()*fps
                                        if post_window > mean_latency-(3*fps):
                                            ax[col].axvline(mean_latency, color='darkorange', alpha=0.5)

                                if epoch == 'response':
                                    ax[col].axvline(pre_window, color='darkorange', alpha=0.5)
                                    if post_window >= fps:
                                        ax[col].axvline(pre_window+fps, color='green', alpha=0.5)

                                if epoch == 'reward':
                                    ax[col].axvline(pre_window, color='green', alpha=0.5)
                                    if pre_window >= fps:
                                        ax[col].axvline(pre_window-fps, color='darkorange', alpha=0.5)

                                if epoch == 'ITI':
                                    ax[col].axvline(pre_window, color='black', alpha=0.5)

                    plt.suptitle((' ').join([roi]), y=1.05)
                    
                    if roi in save_ROIs:
                        fig.savefig(data_directory+'\\'+roi+'_EXAMPLE_traces.svg')


# ## Statistical functions ####

# In[18]:


def std_filter(df,thresh):
    trial_means = df.mean(axis=1).to_frame()
    upper_thresh = trial_means[trial_means > (trial_means.mean() + (thresh*trial_means.std()))].dropna()
    lower_thresh = trial_means[trial_means < (trial_means.mean() - (thresh*trial_means.std()))].dropna()
    pos_df = df[df.index.isin(upper_thresh.index)]
    neg_df = df[df.index.isin(lower_thresh.index)]
    remainder_df = df[~((df.index.isin(upper_thresh.index)) | (df.index.isin(lower_thresh.index)))]
    return {'positive':pos_df, 'negative':neg_df, 'unmodulated':remainder_df}


# In[19]:


def mann_whitney_mean_rank(data1, data2, pair, label, UOA):
    """
    Calculate the Mann-Whitney U mean rank sums and z-statistic.

    Arguments:
    data1 -- List or array of data for group 1
    data2 -- List or array of data for group 2

    Returns:
    u_stat -- U statistic from the Mann-Whitney U test
    p_val -- p-value from the Mann-Whitney U test
    mean_rank_sum1 -- Mean rank sum for group 1
    mean_rank_sum2 -- Mean rank sum for group 2
    z_stat -- Z statistic from the Mann-Whitney U test
    """
    # Check if both columns of data are eligible (i.e. > 0)
    n1 = len(data1)
    n2 = len(data2)
    if (n1 > 0) & (n2 > 0):

        # Combine the data
        combined_data = data1 + data2

        # Calculate the ranks
        ranks = {}
        for i, val in enumerate(sorted(combined_data)):
            if val not in ranks:
                ranks[val] = []
            ranks[val].append(i + 1)

        # Calculate the rank sums for each group
        rank_sum1 = sum(ranks[val][0] for val in data1)
        rank_sum2 = sum(ranks[val][0] for val in data2)

        # Calculate the mean rank sums
        mean_rank_sum1 = rank_sum1 / len(data1)
        mean_rank_sum2 = rank_sum2 / len(data2)

        # Perform Mann-Whitney U test
        u_stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')

        # Calculate the z-statistic
        n_samples = n1 + n2
        z_stat = ((rank_sum1+0.5) - (rank_sum1 + rank_sum2) / 2) / ((n1 * n2 * (n_samples + 1)) / 12) ** 0.5
        column_data = [u_stat, p_val, n1, n2, mean_rank_sum1, mean_rank_sum2, z_stat]
    else:
        column_data = [np.nan,np.nan,np.nan,np.nan,np.nan]
    
    if 'ROI' in UOA:
         index=['U','p-value','nROIs ('+pair[0]+')','nROIs ('+pair[1]+')','mean rank ('+pair[0]+')', 'mean rank ('+pair[1]+')','z-statistic'] 
    else:
         index=['U','p-value','nTrials ('+pair[0]+')','nTrials ('+pair[1]+')','mean rank ('+pair[0]+')', 'mean rank ('+pair[1]+')','z-statistic'] 
    # Create dataframe of results
    results_df = pd.DataFrame(column_data, index=index,columns=[label]).T

    return results_df


# In[20]:


def t_test(pairing, data1, data2, pair, label, UOA):
    """
    Perform a paired t-test and return the results as a Pandas DataFrame.

    Arguments:
    data1 -- List or array of data for group 1
    data2 -- List or array of data for group 2

    Returns:
    result_df -- DataFrame with the p-value, mean, standard error of the mean,
                 and number of samples for each group
    """
    # Check if both columns of data are eligible (i.e. > 0)
    n1 = len(data1)
    n2 = len(data2)
    if (n1 > 0) & (n2 > 0):
        
        # Perform paired t-test
        t_stat, p_val = wilcoxon(data1, data2) if pairing == 'paired' else mannwhitneyu(data1, data2)

        # Calculate mean, standard error of the mean, and number of samples for each group
        mean1 = round(np.mean(data1),3)
        mean2 = round(np.mean(data2),3)
        sem1 = round((np.std(data1, ddof=1) / np.sqrt(len(data1))),3)
        sem2 = round((np.std(data2, ddof=1) / np.sqrt(len(data2))),3)
        column_data = [t_stat, p_val, n1, n2, str(mean1)+' +/- '+str(sem1), str(mean2)+' +/- '+str(sem2)]
    else:
        column_data = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        
    if UOA == 'session_by_session':
         index=['t-stat','p-value','nSessions ('+pair[0]+')','nSessions ('+pair[1]+')','mean +/- SEM ('+pair[0]+')', 'mean +/- SEM ('+pair[1]+')'] 
    else:
         index=['t-stat','p-value','nTrials ('+pair[0]+')','nTrials ('+pair[1]+')','mean +/- SEM ('+pair[0]+')', 'mean +/- SEM ('+pair[1]+')'] 
        
    # Create DataFrame with the results
    result_df = pd.DataFrame(column_data, index=index, columns = [label]).T

    return result_df


# In[21]:


def t_test_pandas(df, comparison_list, min_trials, dtype, UOA, protocols=None, metric=None):
    test_df = {}
    if 'Channel' in df.index.names:
        for channel in df.index.get_level_values('Channel').unique():
            channel_df = df[df.index.get_level_values('Channel') == channel]

            channel_results = {}
            protocol_list = [[x] for x in list(channel_df.index.get_level_values('Protocol').unique())] if protocols == None else protocols
            for protocol in protocol_list:
                protocol_label = protocol[0] if len(protocol) == 1 else ('_').join(protocol)
                protocol_df = channel_df[channel_df.index.get_level_values('Protocol').isin(protocol)]
                if metric == None:
                    metric_df = protocol_df
                    metric_level = 0
                else:
                    metric_df = protocol_df[[metric,'nTrials']]
                    metric_level = 1
                protocol_results = []
                for pair in comparison_list:
                    label = (' vs. ').join([pair[0], pair[1]])
                    
                    pair_list = []
                    for c in pair:
                        c_df = metric_df.T[metric_df.T.index.get_level_values(metric_level) == c].T
                        c_df = c_df[c_df.iloc[:,1] >= min_trials][metric]
                        c = list(c_df.dropna()[c])
                        pair_list.append(c)
                    if dtype == 'events':
                        results_df = mann_whitney_mean_rank(pair_list[0], pair_list[1], pair, label, UOA)
                    else:
                        if UOA == 'session_by_session':
                            results_df = t_test('paired',list(pair_df.iloc[:,0].dropna()), list(pair_df.iloc[:,1].dropna()), pair, label, UOA) 
                        else:
                            results_df = t_test('unpaired',list(pair_df.iloc[:,0].dropna()), list(pair_df.iloc[:,1].dropna()), pair, label, UOA) 
                    protocol_results.append(results_df)
                protocol_results = pd.concat(protocol_results)
                channel_results[protocol_label] = protocol_results
            test_df[channel] = pd.concat(channel_results)


    else:
        channel = 'Behavior'
        channel_df = df

        channel_results = {}
        protocol_list = [[x] for x in list(channel_df.index.get_level_values('Protocol').unique())] if protocols == None else protocols
        for protocol in protocol_list:
            protocol_label = protocol[0] if len(protocol) == 1 else ('_').join(protocol)
            protocol_df = channel_df[channel_df.index.get_level_values('Protocol').isin(protocol)]
            if metric == None:
                metric_df = protocol_df
                metric_level = 0
            else:
                metric_df = protocol_df[metric]
                metric_level = 1
            protocol_results = []
            for pair in comparison_list:
                label = (' vs. ').join([pair[0], pair[1]])
                pair_df = metric_df.T[metric_df.T.index.get_level_values(metric_level).isin(pair)].T
                if UOA == 'session_by_session':
                    pair_df = pair_df.dropna(how='any')
                    results_df = t_test('paired',list(pair_df.iloc[:,0].dropna()), list(pair_df.iloc[:,1].dropna()), pair, label, UOA) 
                else:
                    results_df = t_test('unpaired',list(pair_df.iloc[:,0].dropna()), list(pair_df.iloc[:,1].dropna()), pair, label, UOA) 
                protocol_results.append(results_df)
            protocol_results = pd.concat(protocol_results)
            channel_results[protocol_label] = protocol_results
        test_df[channel] = pd.concat(channel_results)    
    test_df = pd.concat(test_df)
    test_df.index.names = ['Channel','Protocol','Comparison']
    test_df = test_df.sort_values(['Comparison','Protocol'])
#     corrected_pvals = sm.stats.multipletests(list(test_df['p-value']), alpha=0.05, method='bonferroni')
#     test_df['p-value'] = corrected_pvals[1]
    test_df = test_df[test_df['p-value'] < 1]   
    
    return test_df

