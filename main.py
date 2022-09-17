import noise_removal as nr
import utils as ut
import pandas as pd
import numpy as np
from numba import jit
import os
import random
import matplotlib.pyplot as plt
import pywt
from itertools import chain
import math
plt.style.use('seaborn-whitegrid')
from scipy.signal import butter, lfilter, filtfilt
def plot_rpeak_window(df_signal, anno_est, start, size, option1, option2):
    
    if option1 == 'random':
        start = random.randrange(len(df_signal))
    print(start)
    if option2 == 'filtered':
        df_signal = filtered_signal
    
    xs = range(start - size, start + size)
    ys = df_signal[xs] #- np.nanmedian(df_signal[xs])
    values = ['0', '1', '2', '3','4','5', '6', '7', '8', '9', '10'] 
    positions = range(start-3000, start+300, 300)
    
    markers = anno_est[(anno_est['pos'] > xs[0]) & (anno_est['pos'] < xs[len(xs)-1])]['pos']
    markers = np.asarray(markers) - start + size
    plt.rcParams["figure.figsize"] = (14,8)
    plt.figure(dpi=150)

    plt.plot(xs, ys, marker = 'o', color = 'blue', markevery = markers, markerfacecolor='lime', markersize = 8,
            linewidth = 1)
   
    plt.show()
    
def plot_rpeak_window(df_signal, anno_est, start, size, option1, option2):
    
    if option1 == 'random':
        start = random.randrange(len(df_signal))
    print(start)
    if option2 == 'filtered':
        df_signal = filtered_signal
    
    
    xs = range(start - size, start + size)
    ys = df_signal[xs] #- np.nanmedian(df_signal[xs])
    positions = range(start-3000, start+300, 300)
   
    markers = anno_est[(anno_est['pos'] > xs[0]) & (anno_est['pos'] < xs[len(xs)-1])]['pos']
    markers2 = markers.copy()
    ys2 = df_signal[markers2]
    targets = anno_est[(anno_est['pos'] > xs[0]) & (anno_est['pos'] < xs[len(xs)-1])]['target']
    markers = np.asarray(markers) - start + size
    
    plt.rcParams["figure.figsize"] = (14,8)
    plt.figure(dpi=150)

    colors = {'N':'lime', 'S':'purple', 'V':'red'}
    plt.plot(xs, ys, color = 'blue', linewidth = 1)
    plt.scatter(markers2, ys2, c=targets.map(colors), s = 80)
   
    plt.show()
    
def calc_lp(signal, low, high):
    n = len(signal)
    approxi = pywt.downcoef('a', signal, 'db4', level=low)
    baseline = pywt.upcoef('a', approxi, 'db4', level=low, take=n)
    
    approxi = pywt.downcoef('a', signal, 'db4', level=high)
    upper = pywt.upcoef('a', approxi, 'db4', level=high, take=n)
    return upper - baseline


X = nr.QRS_to_Noise('114001_ECG', sRate=250, num_threads=8, 
                path_raw='/ECG/atsense/original',
                path_input='/home/Docker/Python/ECG/dl_inputs',
                path_model='/home/Docker/Python/ECG/models')

X.wfdb_to_npy()

X.save_dl_inputs('114001_ECG.npy')

X.noise_removal('light')

X.qrs_detection()

anno, sig, summary, anno_noise_removed, sig_noise_removed, clean, noise = X.create_output()
