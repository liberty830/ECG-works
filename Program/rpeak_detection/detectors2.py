from IPython.display import clear_output

from ecgdetectors import Detectors
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import glob
import math

from multiprocessing import Pool
import parmap
from datetime import datetime
import wfdb

import re
from numba import jit
import gc
from itertools import chain
import random
from scipy import fftpack
import pywt
import scipy

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from scipy.fft import fft, ifft
from scipy.fft import fftfreq
import sklearn
from utils.util_functions import *

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
    
import scipy.signal as signal
import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

import gc
import sys
import os, psutil


# R-peak detection
def MWA_from_name(function_name):
    if function_name == "cumulative":
        return MWA_cumulative
    elif function_name == "convolve":
        return MWA_convolve
    elif function_name == "original":
        return MWA_original
    else: 
        raise RuntimeError('invalid moving average function!')

@jit(nopython = True)        
def MWA_cumulative(input_array, window_size):
    
    ret = np.cumsum(input_array)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    
    for i in range(1,window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size - 1:]  = ret[window_size - 1:] / window_size

    return ret

@jit(nopython = True)
def panPeakDetect2(detection, sRate):    

    min_distance = int(0.25*sRate)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i>0 and i<len(detection)-1:
            if detection[i-1]<detection[i] and detection[i+1]<detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak]>threshold_I1 and (peak-signal_peaks[-1])>0.3*sRate:
                        
                    signal_peaks.append(peak)
                    indexes.append(index)
                    
                    if (detection[signal_peaks[-1]] > 10*SPKI):
                        SPKI = SPKI
                    else:
                        SPKI = 0.125*detection[signal_peaks[-1]] + 0.875*SPKI
                    if RR_missed!=0:
                        if signal_peaks[-1]-signal_peaks[-2]>RR_missed:
                            missed_section_peaks = peaks[indexes[-2]+1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak-signal_peaks[-2]>min_distance and signal_peaks[-1]-missed_peak>min_distance and detection[missed_peak]>threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2)>0:           
                                missed_peak = missed_section_peaks2[np.argmax(detection[np.array(missed_section_peaks2)])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak   

                else:
                    noise_peaks.append(peak)
                    if (detection[noise_peaks[-1]] < 0.10*NPKI):
                        NPKI = NPKI
                    else:
                        NPKI = 0.125*detection[noise_peaks[-1]] + 0.875*NPKI

                threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
                threshold_I2 = 0.5*threshold_I1

                if len(signal_peaks)>8:
                    RR = np.diff(np.array(signal_peaks[-9:]))
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66*RR_ave)
                
                index = index+1      
    
    signal_peaks.pop(0)

    return signal_peaks

@jit(nopython = True)
def panPeakDetect(ma_sig, sRate):    

    dist_cutoff = int(0.25*sRate)

    real_peaks = [0]
    noise_peaks = []

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    potential_peaks = []
    
    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0
    
    RR_low_limit = 0

    for i in range(len(ma_sig)):

        if (i > 0) & (i < len(ma_sig)-1):
            # i should be larger than its previous and post amplitude for becoming a peak
            if (ma_sig[i-1] < ma_sig[i]) & (ma_sig[i+1] < ma_sig[i]):
                peak = i
                potential_peaks.append(peak)
                
                # The second condition requires two neighboring peaks must be far apart 250ms each other
                if (ma_sig[peak] > threshold_I1) & (peak-real_peaks[-1] > dist_cutoff):
                        
                    real_peaks.append(peak)
                    indexes.append(index)
                    
                     # When the current potential peak did not satisfy the first condition for becoming a peak,
                    # the second condition checks if the one is a peak or not again
                    if RR_missed != 0:
                        # This condition check if there would be a missed interval by calculating RR distance
                        # between two neighboring peaks
                        if real_peaks[-1] - real_peaks[-2] > RR_missed:
                            peaks_in_missed = np.array(potential_peaks[indexes[-2]+1:indexes[-1]])
                            
                            pre = real_peaks[-2]
                            current = real_peaks[-1]
                            condition = (peaks_in_missed-pre > dist_cutoff) & (current-peaks_in_missed > dist_cutoff) & (ma_sig[peaks_in_missed] > threshold_I2)
                            peaks_in_missed2 = peaks_in_missed[np.where(condition)]
                            
                            if len(peaks_in_missed2) > 0:           
                                missed_peak = peaks_in_missed2[np.argmax(ma_sig[peaks_in_missed2])]
                                missed_peaks.append(missed_peak)
                                real_peaks.append(real_peaks[-1])
                                real_peaks[-2] = missed_peak   
                    
                    # This part prevents the outlier peaks from distorting heavily the thresholds
                    # If the current peak is over 10 times the threshold, threshold is not updated
                    if (ma_sig[real_peaks[-1]] > 10*SPKI):
                        SPKI = SPKI
                    else:
                        SPKI = 0.125*ma_sig[real_peaks[-1]] + 0.875*SPKI
                        
                else:
                    noise_peaks.append(peak)
                    # This part prevents the outlier peaks from distorting heavily the thresholds
                    # If the current peak is less than 0.1 times the threshold, threshold is not updated
                    if (ma_sig[noise_peaks[-1]] < 0.10*NPKI):
                        NPKI = NPKI
                    else:
                        NPKI = 0.125*ma_sig[noise_peaks[-1]] + 0.875*NPKI

                threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
                threshold_I2 = 0.5*threshold_I1
                
                # Calculate RR criteria for missed section with 8 most recent peaks 
                
                if len(real_peaks) > 18:
                    
                    if RR_low_limit == 0:
                        RR_default = np.mean(np.diff(np.array(real_peaks[:9])))
                        RR_low_limit = 0.92 * RR_default
                        RR_high_limit = 1.16 * RR_default
                    
                    # Exclude the outlier RR beats for the reference RR to find the missed sections
                    RR_interval = np.diff(np.array(real_peaks[-18:]))
                    acceptable_beats = np.where((RR_interval > RR_low_limit) & (RR_interval < RR_high_limit))[0]
                    
                    if len(acceptable_beats) > 8:
                        RR_ave = int(np.mean(RR_interval[acceptable_beats][-9:]))
                    elif (len(acceptable_beats) > 4) & (len(acceptable_beats) <= 8):
                        RR_ave = int(np.mean(RR_interval[acceptable_beats]))
                    else:
                        RR_ave = int(np.mean(RR_interval[-9:]))
                    
                    RR_missed = int(1.66*RR_ave)
                    RR_low_limit = 0.92 * RR_ave
                    RR_high_limit = 1.16 * RR_ave
                
                index = index+1      
    
    real_peaks.pop(0)

    return real_peaks


def pan_tompkins_detector(filtered_ecg, sRate, low_pass, high_pass, MWA_name='cumulative'):
        
        filtered_ecg = calc_lp(filtered_ecg, low_pass, high_pass)
        diff = np.diff(filtered_ecg) 
        diff = np.append(diff[0], diff)
        squared = diff*diff
        
        N = int(0.12*sRate)
        mwa = MWA_from_name(MWA_name)(squared, N)
        mwa[:int(0.2*sRate)] = 0

        mwa_peaks = panPeakDetect(mwa, sRate)
        
        return mwa_peaks

@jit(nopython = True)
def two_average_qrs(filtered_ecg, start, end, sRate):
    QRS = []
    for i in range(len(start)):
        if end[i] - start[i] > int(0.08*sRate):
            detection = np.argmax(filtered_ecg[start[i]:end[i]+1])+start[i]
            if QRS:
                if detection - QRS[-1] > int(0.3*sRate):
                    QRS.append(detection)
            else:
                QRS.append(detection)
    
    return QRS

def two_average_detector(filtered_ecg, sRate, low_pass, high_pass, MWA_name='cumulative'):
    
    filtered_ecg= calc_lp(filtered_ecg, 4, 3)
 
    diff = np.diff(filtered_ecg) 
    diff = np.append(diff[0], diff)
    squared = diff*diff
    
    diff = np.diff(squared) 
    diff = np.append(diff[0], diff)
    squared = diff*diff
       
    window1 = int(0.12*sRate)
    mwa_qrs2 = MWA_from_name(MWA_name)(abs(squared), window1)
    mwa_qrs = MWA_from_name(MWA_name)(abs(filtered_ecg), window1)

    window2 = int(0.6*sRate)
    mwa_beat2 = MWA_from_name(MWA_name)(abs(squared), window2)
    mwa_beat = MWA_from_name(MWA_name)(abs(filtered_ecg), window2)

    blocks = np.zeros(len(filtered_ecg))
    block_height = np.nanmax(filtered_ecg)

    blocks[np.where((mwa_qrs > mwa_beat) & (mwa_qrs2 > mwa_beat2))] = block_height
    lead_first = blocks[0]
    lead_last = blocks[-1]
    blocks_lead = np.roll(blocks, 1)
    blocks_lead[0] = lead_first
    blocks_lead[-1] = lead_last
    block_height_col = np.full(blocks.shape, block_height)

    block_matrix = np.c_[blocks_lead, blocks]
    start = list(np.where(block_matrix[:,0] < block_matrix[:,1]))
    start = start[0]
    start = start[:-2]
    end = list(np.where(block_matrix[:,0] > block_matrix[:,1]))
    end = end[0]-1

    QRS = two_average_qrs(filtered_ecg, start, end, sRate)

    return QRS


def fix_wide_rr(pan, elgendi, cutoff_rr):
    pan = np.array(pan)
    elgendi = np.array(elgendi)
    
    if len(pan > 0):
        pan_lag = np.roll(pan, 1)
        pan_lag[0] = 0
        rr_pre = pan - pan_lag # This is RR interval
        location_post = np.array(np.where(rr_pre > cutoff_rr)[0])
        location_pre = location_post - 1

        A = [np.where((pan[location_pre[i]] < elgendi) & (pan[location_post[i]] > elgendi))[0] 
             for i in range(len(location_post))] 
        
        # "supplements" are the location of R-peaks that Pan missed & Elgendi found
        supplements = np.array([item for l in  A for item in l])
        if len(supplements) > 0:
            rpeaks = np.concatenate((pan, elgendi[supplements]))
            rpeaks = list(set(rpeaks))
            rpeaks.sort()
        else:
            rpeaks = pan
    else:
        rpeaks = elgendi

    return rpeaks  

def pan_elgendi(band_signal, sRate, cutoff_wide):
    
    # Apply Pan and Elgendi algorithms to the band_passed signal
    pan = pan_tompkins_detector(band_signal, sRate = sRate, low_pass=4, high_pass=3)
    gc.collect()
    elgendi = two_average_detector(band_signal, sRate = sRate, low_pass=4, high_pass=3)
    gc.collect()
    # If Pan missed the R-peaks, we supplement these with the R-peaks which were found by Elgendi
    # "Cutoff_wide" is the criteria to decide missing R-peaks 
    if len(pan) > 0 or len(elgendi) > 0:
        rpeak = fix_wide_rr(pan, elgendi, cutoff_wide) 
    else:
        rpeak = []
    return np.array(rpeak)

def qrs_first_round2(raw_signal, sRate, cutoff_wide, search_window, num_threads):
 
    # Split the raw signal into the number of threads for parallel computing
    # Each splitted signal should be greater than 100000 samples to find R-peaks
    x = int(len(raw_signal)/100000)
    num_threads = min(num_threads, x)
    cutoff = np.linspace(1, len(raw_signal), num_threads, dtype = int)
    cutoff = np.delete(cutoff, [0, len(cutoff)-1])
    
    splitted_signal = np.split(raw_signal, cutoff)

    # Arguments for parallel computing
    input_list = [(splitted_signal[i], sRate, cutoff_wide) 
                  for i in range(len(splitted_signal))]
    
    with Pool(num_threads) as pool:
        x = pool.starmap(pan_elgendi, input_list)

    # The starting location of each splitted signal should be added
    y = [x[w] + cutoff[w-1] for w in range(len(x)) if w > 0]
    y = np.concatenate(y, axis = 0)
    rpeaks = np.concatenate((x[0], y), axis = 0)

    return rpeaks


def qrs_first_round(raw_signal, sRate, cutoff_wide, search_window, num_threads):
    print('Start First Round!')
    # Split the raw signal into the number of threads for parallel computing
    # Each splitted signal should be greater than 100000 samples to find R-peaks
    x = int(len(raw_signal)/100000)
    num_threads = 23
    cutoff = np.linspace(1, len(raw_signal), num_threads, dtype = int)
    cutoff = np.delete(cutoff, [0, len(cutoff)-1])
   
    splitted_signal = np.split(raw_signal, cutoff)
  
    del raw_signal
    gc.collect()
   
    x = []
    for i in range(22):
        x.append(pan_elgendi(splitted_signal[i], sRate, cutoff_wide))

    # The starting location of each splitted signal should be added
    y = [x[w] + cutoff[w-1] for w in range(len(x)) if w > 0]
    y = np.concatenate(y, axis = 0)
    rpeaks = np.concatenate((x[0], y), axis = 0)

    return rpeaks



# This process is the second round to find R-peaks
# If there are long intervals with missed R-peaks, we find the R-peaks within those intervals
def qrs_second_round(raw_signal, rpeaks, sRate, cutoff_wide, search_window):
    print('Start Second Round!')
    all_peaks = rpeaks.tolist()
    sub_signals = []
    
    for i in range(0, 5):
        all_peaks.sort()
        anno_est = np.array(all_peaks, dtype = np.int32)
        x = anno_est - np.roll(anno_est, 1) 
        x[0] = 300
        wide_end = anno_est[np.where(x > sRate*3)]
        wide_start = anno_est[np.where(x > sRate*3)[0] - 1]
        
        rpeak_big_array = []
        for j in range(len(wide_start)):
            signal_sub = np.array(raw_signal[wide_start[j]:wide_end[j]])
            sub_signals.append(signal_sub)
            rpeak = pan_elgendi(signal_sub, sRate, cutoff_wide)

            if len(rpeak) > 0:
                rpeak = rpeak + wide_start[j]
                rpeak_big_array.append(rpeak)

        missing = list(chain(*rpeak_big_array))
        
        if len(missing) > 0:
            all_peaks = all_peaks + missing
    all_peaks.sort()

    return all_peaks

# This moves the approximate R-peaks to the correct location
def correct_peak(filtered_signal, qrs_array, sRate, size):
    peaks = np.array(qrs_array)
    peaks = peaks[np.where(peaks > size)]
    
    ecg_big_array = cut_signal(filtered_signal, peaks, size)
    max_per_window = [np.argmax(w) for w in ecg_big_array]
    max_per_window = np.asarray(max_per_window)
    addings = peaks[:]
    addings = [w-size for w in addings]
    addings = np.asarray(addings)
    max_per_window = np.add(max_per_window, addings)

    return max_per_window

# This removes false R-peaks by its amplitude of slope
def remove_too_small(filtered_signal, rpeaks_array, cutoff_peak, sRate):

    filtered_signal  = np.absolute(filtered_signal) 
    too_small = np.where(filtered_signal[rpeaks_array] < cutoff_peak)[0]
    rpeak_corrected = np.delete(rpeaks_array, too_small)
    
    return rpeak_corrected

# This chooses the right R-peak between two R-peaks that are located too closely
def correct_too_narrow(ecg1, rpeaks_array, sRate):

    rpeak_lag = np.roll(rpeaks_array, 1)
    
    rpeak_lag[0] = 0
    rr_pre = rpeaks_array - rpeak_lag
    location_after = np.array(np.where(rr_pre < sRate*0.20)[0])
    location_before = location_after - 1
    rpeak_before = rpeaks_array[location_before]
    rpeak_after = rpeaks_array[location_after]
    ecg1 = np.diff(ecg1)
    ecg1 = np.absolute(ecg1)

    if len(rpeak_before) > 0:
        
        ecg_narrow_big_array1 = np.asarray([np.nanmean(ecg1[x-2:x+2]) for x in rpeak_before])
        ecg_narrow_big_array2 = np.asarray([np.nanmean(ecg1[x-2:x+2]) for x in rpeak_after])
        
        remove2 =  rpeak_after[ecg_narrow_big_array1 - ecg_narrow_big_array2 > 0]
        remove1 =  rpeak_before[ecg_narrow_big_array1 - ecg_narrow_big_array2 < 0]

        rpeak_remove = np.concatenate((remove1, remove2))
        rpeak_corrected = np.setdiff1d(rpeaks_array,rpeak_remove)
    else:
        rpeak_corrected = rpeaks_array
    
    return rpeak_corrected

# Main function
def rpeak_final(raw_signal, filtered_signal, sRate, cutoff_wide, 
                cutoff_height, search_window, search_window2, num_threads):
    process = psutil.Process(os.getpid())
    print(str(process.memory_info().rss) + 'A')   
    diff = np.diff(filtered_signal)
    squared = diff*diff
    diff = np.diff(squared)
    process = psutil.Process(os.getpid())
    print(str(process.memory_info().rss) + 'B')   
    rpeaks = qrs_first_round(raw_signal, sRate, cutoff_wide, search_window, num_threads)
    rpeaks = qrs_second_round(raw_signal, rpeaks, sRate,  cutoff_wide, search_window)
    del raw_signal
    rpeaks = correct_peak(diff*diff, rpeaks, sRate, search_window2)
    del diff
    gc.collect()
    process = psutil.Process(os.getpid())
    print(str(process.memory_info().rss) + 'C')   
    rpeaks = remove_too_small(squared, rpeaks, cutoff_height, sRate)
    rnew = correct_too_narrow(filtered_signal, rpeaks, sRate)
    del squared, filtered_signal
    gc.collect()
    process = psutil.Process(os.getpid())
    print(str(process.memory_info().rss) + 'D')   
#     anno_est = pd.DataFrame({'loc': rnew, 'target': 'R'})
#     anno_est['rr_pre'] = anno_est['loc'] - np.roll(anno_est['loc'], 1)
#     anno_est['rr_pre'][0] = 300
#     anno_est['rr_post'] = np.roll(anno_est['loc'], -1) - anno_est['loc'] 
#     anno_est['rr_post'][len(anno_est)-1] = 300
#     anno_est['rr_ratio'] = anno_est['rr_pre'] / anno_est['rr_post']
    
    return rnew














@jit(nopython = True)        
def running_mean(x, N):
    
    cumsum = np.cumsum(x) 
    cumsum[N:] = (cumsum[N:] - cumsum[:-N]) / float(N)
    
    for i in range(1, N):
        cumsum[i-1] = cumsum[i-1] / i

    return cumsum

@jit(nopython = True)
def pan_tompkins_peak(ma_sig, sRate, option):    

    dist_cutoff = int(0.25*sRate)

    real_peaks = [0]
    noise_peaks = []

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    potential_peaks = []
    
    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0
    
    RR_low_limit = 0

    for i in range(len(ma_sig)):

        if (i > 0) & (i < len(ma_sig)-1):
            # i should be larger than its previous and post amplitude for becoming a peak
            if (ma_sig[i-1] < ma_sig[i]) & (ma_sig[i+1] < ma_sig[i]):
                peak = i
                potential_peaks.append(peak)
                
                # The second condition requires two neighboring peaks must be far apart 250ms each other
                if (ma_sig[peak] > threshold_I1) & (peak-real_peaks[-1] > dist_cutoff):
                        
                    real_peaks.append(peak)
                    indexes.append(index)
                    
                     # When the current potential peak did not satisfy the first condition for becoming a peak,
                    # the second condition checks if the one is a peak or not again
                    if RR_missed != 0:
                        # This condition check if there would be a missed interval by calculating RR distance
                        # between two neighboring peaks
                        if real_peaks[-1] - real_peaks[-2] > RR_missed:
                            peaks_in_missed = np.array(potential_peaks[indexes[-2]+1:indexes[-1]])
                            
                            pre = real_peaks[-2]
                            current = real_peaks[-1]
                            condition = (peaks_in_missed-pre > dist_cutoff) & (current-peaks_in_missed > dist_cutoff) & (ma_sig[peaks_in_missed] > threshold_I2)
                            peaks_in_missed2 = peaks_in_missed[np.where(condition)]
                            
                            if len(peaks_in_missed2) > 0:           
                                missed_peak = peaks_in_missed2[np.argmax(ma_sig[peaks_in_missed2])]
                                missed_peaks.append(missed_peak)
                                real_peaks.append(real_peaks[-1])
                                real_peaks[-2] = missed_peak   
                    
                    # This part prevents the outlier peaks from distorting heavily the thresholds
                    # If the current peak is over 10 times the threshold, threshold is not updated
                    if (ma_sig[real_peaks[-1]] > 10*SPKI):
                        SPKI = SPKI
                    else:
                        SPKI = 0.125*ma_sig[real_peaks[-1]] + 0.875*SPKI
                        
                else:
                    noise_peaks.append(peak)
                    # This part prevents the outlier peaks from distorting heavily the thresholds
                    # If the current peak is less than 0.1 times the threshold, threshold is not updated
                    if (ma_sig[noise_peaks[-1]] < 0.10*NPKI):
                        NPKI = NPKI
                    else:
                        NPKI = 0.125*ma_sig[noise_peaks[-1]] + 0.875*NPKI

                threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
                threshold_I2 = 0.5*threshold_I1
                   
                if option == 'origianl':
                    if len(real_peaks) > 18:

                        if RR_low_limit == 0:
                            RR_default = np.mean(np.diff(np.array(real_peaks[:9])))
                            RR_low_limit = 0.92 * RR_default
                            RR_high_limit = 1.16 * RR_default

                        # Exclude the outlier RR beats for the reference RR to find the missed sections
                        RR_interval = np.diff(np.array(real_peaks[-18:]))
                        acceptable_beats = np.where((RR_interval > RR_low_limit) & (RR_interval < RR_high_limit))[0]

                        if len(acceptable_beats) > 8:
                            RR_ave = int(np.mean(RR_interval[acceptable_beats][-9:]))
                        elif (len(acceptable_beats) > 4) & (len(acceptable_beats) <= 8):
                            RR_ave = int(np.mean(RR_interval[acceptable_beats]))
                        else:
                            RR_ave = int(np.mean(RR_interval[-9:]))

                        RR_missed = int(1.66*RR_ave)
                        RR_low_limit = 0.92 * RR_ave
                        RR_high_limit = 1.16 * RR_ave
                else:
                    if len(real_peaks) > 8:
                        RR_interval = np.diff(np.array(real_peaks[-9:]))
                        RR_ave = int(np.mean(RR_interval))
                        RR_missed = int(1.66*RR_ave)
                
                index = index+1      
    
    real_peaks.pop(0)

    return real_peaks    

def pan_tompkins_detector(filtered_ecg, option, sRate, low_pass, high_pass):
        
        filtered_ecg = calc_lp(filtered_ecg, 4, 3)
        diff = np.diff(filtered_ecg) 
        diff = np.append(diff[0], diff)
        squared = diff*diff
        
        N = int(0.12*sRate)
        units = running_mean(squared, N)
        units[:int(0.2*sRate)] = 0

        peaks_array = pan_tompkins_peak(units, sRate, option)

        return peaks_array
    
@jit(nopython = True)
def elgendi_peak(filtered_ecg, start, end, sRate):
    qrs_array = []
    for i in range(len(start)):
        if end[i] - start[i] > int(0.08*sRate):
            potential_peak = np.argmax(filtered_ecg[start[i]:end[i]+1])+start[i]
            if qrs_array:
                if potential_peak - qrs_array[-1] > int(0.3*sRate):
                    qrs_array.append(potential_peak)
            else:
                qrs_array.append(potential_peak)
    return qrs_array

def elgendi_detector(filtered_ecg, sRate, low_pass, high_pass):
    
    filtered_ecg= calc_lp(filtered_ecg, 4, 3)
 
    diff = np.diff(filtered_ecg) 
    diff = np.append(diff[0], diff)
    squared = diff*diff
    
    diff = np.diff(squared) 
    diff = np.append(diff[0], diff)
    squared = diff*diff
       
    short_range = int(0.12*sRate)
    mwa_short2 = running_mean(abs(squared), short_range)
    mwa_short = running_mean(abs(filtered_ecg), short_range)

    long_range = int(0.6*sRate)
    mwa_long2 = running_mean(abs(squared), long_range)
    mwa_long = running_mean(abs(filtered_ecg), long_range)

    blocks = np.zeros(len(filtered_ecg))
    block_max = np.nanmax(filtered_ecg)

    blocks[np.where((mwa_short > mwa_long) & (mwa_short2 > mwa_long2))] = block_max
    lead_first = blocks[0]
    lead_last = blocks[-1]
    blocks_lead = np.roll(blocks, 1)
    blocks_lead[0] = lead_first
    blocks_lead[-1] = lead_last
    block_max_col = np.full(blocks.shape, block_max)

    block_matrix = np.c_[blocks_lead, blocks]
    start = list(np.where(block_matrix[:,0] < block_matrix[:,1]))
    start = start[0]
    start = start[:-2]
    end = list(np.where(block_matrix[:,0] > block_matrix[:,1]))
    end = end[0]-1

    QRS = elgendi_peak(filtered_ecg, start, end, sRate)

    return QRS



def fix_wide_rr(pan, elgendi, cutoff_rr):
    pan = np.array(pan)
    elgendi = np.array(elgendi)
    
    if len(pan > 0):
        pan_lag = np.roll(pan, 1)
        pan_lag[0] = 0
        rr_pre = pan - pan_lag # This is RR interval
        location_post = np.array(np.where(rr_pre > cutoff_rr)[0])
        location_pre = location_post - 1

        A = [np.where((pan[location_pre[i]] < elgendi) & (pan[location_post[i]] > elgendi))[0] 
             for i in range(len(location_post))] 
        
        # "supplements" are the location of R-peaks that Pan missed & Elgendi found
        supplements = np.array([item for l in  A for item in l])
        if len(supplements) > 0:
            rpeaks = np.concatenate((pan, elgendi[supplements]))
            rpeaks = list(set(rpeaks))
            rpeaks.sort()
        else:
            rpeaks = pan
    else:
        rpeaks = elgendi

    return rpeaks  

def pan_elgendi(band_signal, sRate, cutoff_wide, option):
    
    # Apply Pan and Elgendi algorithms to the band_passed signal
    pan = pan_tompkins_detector(band_signal, option, sRate = sRate, low_pass=4, high_pass=3)
    elgendi = elgendi_detector(band_signal, sRate = sRate, low_pass=4, high_pass=3)
    
    # If Pan missed the R-peaks, we supplement these with the R-peaks which were found by Elgendi
    # "Cutoff_wide" is the criteria to decide missing R-peaks 
    if len(pan) > 0 or len(elgendi) > 0:
        rpeak = fix_wide_rr(pan, elgendi, cutoff_wide) 
    else:
        rpeak = []
    return np.array(rpeak)

# This process is the second round to find R-peaks
# If there are long intervals with missed R-peaks, we find the R-peaks within those intervals
def qrs_first_round(raw_signal, sRate, cutoff_wide, num_threads, option):
    print('Start First Round!')
    # Split the raw signal into the number of threads for parallel computing
    # Each splitted signal should be greater than 100000 samples to find R-peaks
    x = int(len(raw_signal)/100000)
    cutoff = np.linspace(1, len(raw_signal), num_threads, dtype = int)
    cutoff = np.delete(cutoff, [0, len(cutoff)-1])
   
    splitted_signal = np.split(raw_signal, cutoff)
   
    x = []
    for i in range(len(splitted_signal)):
        x.append(pan_elgendi(splitted_signal[i], sRate, cutoff_wide, option))

    # The starting location of each splitted signal should be added
    y = [x[w] + cutoff[w-1] for w in range(len(x)) if w > 0]
    y = np.concatenate(y, axis = 0)
    rpeaks = np.concatenate((x[0], y), axis = 0)

#     rpeaks = pan_elgendi(raw_signal, sRate, cutoff_wide)

    return rpeaks

def qrs_second_round(raw_signal, rpeaks, sRate, cutoff_wide, option):
    print('Start Second Round!')
    all_peaks = rpeaks.tolist()
    sub_signals = []
    
    for i in range(0, 5):
        all_peaks.sort()
        anno_est = np.array(all_peaks, dtype = np.int32)
        x = anno_est - np.roll(anno_est, 1) 
        x[0] = 300
        wide_end = anno_est[np.where(x > sRate*3)]
        wide_start = anno_est[np.where(x > sRate*3)[0] - 1]
        
        rpeak_big_array = []
        for j in range(len(wide_start)):
            signal_sub = np.array(raw_signal[wide_start[j]:wide_end[j]])
            sub_signals.append(signal_sub)
            rpeak = pan_elgendi(signal_sub, sRate, cutoff_wide, option)

            if len(rpeak) > 0:
                rpeak = rpeak + wide_start[j]
                rpeak_big_array.append(rpeak)

        missing = list(chain(*rpeak_big_array))
        
        if len(missing) > 0:
            all_peaks = all_peaks + missing
    all_peaks.sort()

    return all_peaks
# This moves the approximate R-peaks to the correct location
def correct_peak(filtered_signal, qrs_array, sRate, size):
    peaks = np.array(qrs_array)
    peaks = peaks[np.where(peaks > size)]
    
    ecg_big_array = cut_signal(filtered_signal, peaks, size)
    max_per_window = [np.argmax(w) for w in ecg_big_array]
    max_per_window = np.asarray(max_per_window)
    addings = peaks[:]
    addings = [w-size for w in addings]
    addings = np.asarray(addings)
    max_per_window = np.add(max_per_window, addings)

    return max_per_window

# This removes false R-peaks by its amplitude of slope
def remove_too_small(filtered_signal, rpeaks_array, cutoff_peak, sRate):

    filtered_signal  = np.absolute(filtered_signal) 
    too_small = np.where(filtered_signal[rpeaks_array] < cutoff_peak)[0]
    rpeak_corrected = np.delete(rpeaks_array, too_small)
    
    return rpeak_corrected

def correct_too_narrow(ecg1, rpeaks_array, sRate):

    rpeak_lag = np.roll(rpeaks_array, 1)
    
    rpeak_lag[0] = 0
    rr_pre = rpeaks_array - rpeak_lag
    location_after = np.array(np.where(rr_pre < sRate*0.20)[0])
    location_before = location_after - 1
    rpeak_before = rpeaks_array[location_before]
    rpeak_after = rpeaks_array[location_after]
    ecg1 = np.diff(ecg1)
    ecg1 = np.absolute(ecg1)

    if len(rpeak_before) > 0:
        
        ecg_narrow_big_array1 = np.asarray([np.nanmean(ecg1[x-2:x+2]) for x in rpeak_before])
        ecg_narrow_big_array2 = np.asarray([np.nanmean(ecg1[x-2:x+2]) for x in rpeak_after])
        
        remove2 =  rpeak_after[ecg_narrow_big_array1 - ecg_narrow_big_array2 > 0]
        remove1 =  rpeak_before[ecg_narrow_big_array1 - ecg_narrow_big_array2 < 0]

        rpeak_remove = np.concatenate((remove1, remove2))
        rpeak_corrected = np.setdiff1d(rpeaks_array,rpeak_remove)
    else:
        rpeak_corrected = rpeaks_array
    
    return rpeak_corrected

# Main function
def rpeak_final(raw_signal, filtered_signal, sRate, cutoff_wide, 
                cutoff_height, search_window2, num_threads, option):
    
    process = psutil.Process(os.getpid())
    print(str(process.memory_info().rss) + 'A')   
    diff = np.diff(filtered_signal)
    squared = diff*diff
    diff = np.diff(squared)
    squared2 = diff*diff
    
    rpeaks = qrs_first_round(raw_signal, sRate, cutoff_wide, num_threads, option)
    rpeaks = qrs_second_round(raw_signal, rpeaks, sRate,  cutoff_wide, option)
    del raw_signal
    rpeaks = correct_peak(squared2, rpeaks, sRate, search_window2)
    del diff
    gc.collect()
    process = psutil.Process(os.getpid())
    print(str(process.memory_info().rss) + 'C')   
    rpeaks = remove_too_small(squared, rpeaks, cutoff_height, sRate)
    rnew = correct_too_narrow(filtered_signal, rpeaks, sRate)
    del squared, filtered_signal
    gc.collect()
    process = psutil.Process(os.getpid())
    print(str(process.memory_info().rss) + 'D')  

    return rnew

def calc_lp(signal, low, high):
    n = len(signal)
    approxi = pywt.downcoef('a', signal, 'db4', level=low)
    baseline = pywt.upcoef('a', approxi, 'db4', level=low, take=n)
    
    approxi = pywt.downcoef('a', signal, 'db4', level=high)
    upper = pywt.upcoef('a', approxi, 'db4', level=high, take=n)
    return upper - baseline
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


