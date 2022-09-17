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

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
    
import scipy.signal as signal
import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

# Utility functions
@jit(nopython=True)
def count_sign_changes_array(signal_arrays):
    
    counts = []
    np.where(np.diff(np.sign(signal_arrays[0])))[0]
    for w in signal_arrays:
        x = np.where(np.diff(np.sign(w)))[0]
        counts.append(len(x))
    return np.asarray(counts)

def normalize(x):
    y = x/(np.nanmax(x) - np.nanmin(x))
    return y

def normalize_to_range(x, a, b):
    
    y = (b-a)*(x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)) + a
    return y

def normalize_to_range_2d(signal_array):
        
    x = []
    for i in range(0, signal_array.shape[0]):
        x.append(normalize(signal_array[i,:]))
    
    x = np.array(x)
    return x

def calc_lp(signal, low, high):
    n = len(signal)
    approxi = pywt.downcoef('a', signal, 'db4', level=low)
    baseline = pywt.upcoef('a', approxi, 'db4', level=low, take=n)
    
    approxi = pywt.downcoef('a', signal, 'db4', level=high)
    upper = pywt.upcoef('a', approxi, 'db4', level=high, take=n)
    return upper - baseline

@jit(nopython = True)
def cut_signal(ecg1, centers, size):
    ecg_big_array = [ecg1[x-size:x+size] for x in centers]
    return ecg_big_array

@jit(nopython = True)
def get_sum_pre_noise(anno):
    clean_window_idx = np.where(anno == 0)[0]
    sum_pre_noise = [np.sum(anno[:w])*900 for w in clean_window_idx]
    return sum_pre_noise

def get_summary_anno(x, y):
    x = np.asarray(x)
    X = pd.DataFrame({
                      'noise_ratio(%)': y,
                      'num_QRS' : len(x),
                      'mean': np.nanmean(x),
                      'p_50': np.nanquantile(x, 0.50),
                      'max': np.nanmax(x),
                      'sd': np.nanstd(x), 
                      'num_wide_rr': len(np.where(x > 600)[0]),
                      'num_narrow_rr': len(np.where(x < 60)[0])}, index = [0])
    return(X.round(2))