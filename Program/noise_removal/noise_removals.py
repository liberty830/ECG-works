# Basic libraries
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# Signal processing
import wfdb
import scipy.signal as signal
from scipy import fftpack
import pywt
import scipy
from ecgdetectors import Detectors


# ETC
from numba import jit
from itertools import chain
import random
import time
from IPython.display import clear_output
import gc
import sys

# Multi-processing
from multiprocessing import Pool
import parmap
from multiprocessing import cpu_count

# Modeling and deep learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from scipy.fft import fft, ifft
from scipy.fft import fftfreq
import sklearn
from scipy.signal import butter, lfilter, filtfilt
# User-defined libraries
from utils.util_functions import *
from rpeak_detection.detectors import *
import os, psutil



# Data processing main function
def create_input_900(file_name, patient_id, path_raw, path):

    os.chdir(path_raw)
    df_signal = np.load(file_name)
    df_signal = df_signal[:,0]

    os.chdir(path)
    rest = (len(df_signal) % 900)
    if rest != 0:
        y = df_signal[:-rest]
    else:
        y = df_signal
    y = np.reshape(y, (int(len(y)/900), 900))
    ffts = np.absolute(fft(y))
    y = np.expand_dims(y, 1)
    ffts = np.expand_dims(ffts, 1)

    np.save(patient_id + '_signal_900.npy', y)
    np.save(patient_id + '_fft_900.npy', ffts)

    
def create_input_750(file_name, patient_id, path_raw, path):

    os.chdir(path_raw)
    df_signal = np.load(file_name)
    df_signal = df_signal[:,0]

    os.chdir(path)
    rest = (len(df_signal) % 750)
    if rest != 0:
        y = df_signal[:-rest]
    else:
        y = df_signal
    y = np.reshape(y, (int(len(y)/750), 750))
    ffts = np.absolute(fft(y))
    y = np.expand_dims(y, 1)
    ffts = np.expand_dims(ffts, 1)

    np.save(patient_id + '_signal_750.npy', y)
    np.save(patient_id + '_fft_750.npy', ffts)


@jit(nopython = True)
def normalization(signal) :
    for i in range(len(signal)):
        signal[i] = (signal[i] - np.min(signal[i])) / (np.max(signal[i]) - np.min(signal[i]))
    return signal
    



# Modeling and prediction

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)

class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)
    
class Classifier_900(nn.Module):
    def __init__(self, raw_ni, fft_ni, no, drop=.5):
        super().__init__()

        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(1792, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.fft = nn.Sequential(
            SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
            SepConv1d(    32,  64, 8, 2, 4, drop=drop),
            SepConv1d(    64, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 256, 8, 2, 3),
            Flatten(),
            nn.Dropout(drop), nn.Linear(1792, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))
        
    def forward(self, t_raw, t_fft):
    
        raw_out = self.raw(t_raw)
        #print("raw_out", raw_out.shape)
        fft_out = self.fft(t_fft)
        #print("fft_out", fft_out.shape)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(t_in)
        return out
    
class Classifier_750(nn.Module):
    def __init__(self, raw_ni, fft_ni, no, drop=.5):
        super().__init__()

        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(1280, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.fft = nn.Sequential(
            SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
            SepConv1d(    32,  64, 8, 2, 4, drop=drop),
            SepConv1d(    64, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 256, 8, 2, 3),
            Flatten(),
            nn.Dropout(drop), nn.Linear(1536, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))
        
    def forward(self, t_raw, t_fft):
    
        raw_out = self.raw(t_raw)
        #print("raw_out", raw_out.shape)
        fft_out = self.fft(t_fft)
        #print("fft_out", fft_out.shape)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(t_in)
        return out

def create_datasets(data, target, train_size, valid_pct=0.2, seed=None):

    raw, fft = data
    raw[np.isnan(raw)] = 0
    fft[np.isnan(fft)] = 0
    assert len(raw) == len(fft)
    sz = train_size
    idx = np.arange(sz)
    trn_idx, val_idx = train_test_split(
        idx, test_size=valid_pct, random_state=seed)
    trn_ds = TensorDataset(
        torch.tensor(raw[:sz][trn_idx]).float(), 
        torch.tensor(fft[:sz][trn_idx]).float(), 
        torch.tensor(target[:sz][trn_idx]).long())
    val_ds = TensorDataset(
        torch.tensor(raw[:sz][val_idx]).float(), 
        torch.tensor(fft[:sz][val_idx]).float(), 
        torch.tensor(target[:sz][val_idx]).long())
    tst_ds = TensorDataset(
        torch.tensor(raw[sz:]).float(), 
        torch.tensor(fft[sz:]).float(), 
        torch.tensor(target[sz:]).long())
    return trn_ds, val_ds, tst_ds

def create_loaders(data, bs=128, jobs=0):
    trn_ds, val_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return trn_dl, val_dl, tst_dl

def create_testset(data, target):

    raw, fft = data
    raw[np.isnan(raw)] = 0
    fft[np.isnan(fft)] = 0
    assert len(raw) == len(fft)

    tst_ds = TensorDataset(
        torch.tensor(raw).float(), 
        torch.tensor(fft).float(), 
        torch.tensor(target).long())
    return tst_ds

def create_loaders_test(data, bs=128, jobs=0):
    tst_ds = data
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return tst_dl    
    
def process_prediction(sRate, sig, fft, model_name, hours, path_input, path_model):
    
    # input 
    os.chdir(path_input)
    test_org = np.load(sig)
    test_fft_org = np.load(fft)
  
    test = test_org[0:1200*hours] +0
    test_fft = test_fft_org[0:1200*hours] +0 
    del test_org, test_fft_org

    target_te = np.zeros(len(test))

    sign_change_idx = np.where(count_sign_changes_array(test)/int(3*sRate) > 0.075)[0]

    tst_dl = create_testset((test, test_fft), target_te)
    tst_dl = create_loaders_test(tst_dl, bs=256)
    test_fft = 0
   
    del test_fft
    gc.collect()
    
    # Load the saved model

    os.chdir(path_model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if sRate == 300:
        model = Classifier_900(1, 1, 2).to(device)
    elif sRate == 250:
        model = Classifier_750(1, 1, 2).to(device)
    else:
        print('sRate must be 250 or 300!')
        
    state_dict = torch.load(model_name, map_location='cuda:0')
    model.load_state_dict(state_dict)
#     model.load_state_dict(torch.load(model_name))
    print(sys.getsizeof(model))
    print(sys.getsizeof(state_dict))
    model.eval()
    
    # Prediction process

    pred = []
    correct, total = 0, 0
    true = []
    test_signal = []
    probs = []

    for batch in tst_dl:
        x_raw, x_fft, y_batch = [t.to(device) for t in batch]
        out = model(x_raw, x_fft)
#         prob = F.softmax(out, dim=1)
#         prob = prob[:,0].detach().cpu().numpy()
        preds =  F.softmax(out, dim=1).argmax(dim=1)
#         probs.append(prob)
        pred.append(preds.tolist())
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()
    

    pred = np.asarray(list(chain.from_iterable(pred)))
    flag = np.max(np.absolute(test), axis = 2).squeeze()
    noise_idx = np.where((flag < 0.075) | (flag > 1.5))[0]
    pred[noise_idx] = 1
    pred[sign_change_idx] = 1
#     probs = np.asarray(list(chain.from_iterable(probs)))
    del model, state_dict
    gc.collect()
    idx = np.where(np.asarray(pred) == 0)[0]
#     clean_signal = test[idx,:, :]
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 
    return pred, test

def create_fake_noise(sRate, signal_array, num_z):
    
    num1 = int(len(signal_array)/2)
    num2 = int(len(signal_array)/4)
    signal_array = signal_array[0:(num1+2*num2),:,:]
    
    fake_signals1 = []
    fake_signals2 = []
    highs = random.choices(range(6, 9), weights=[0.5, 0.25, 0.25],  k = num1)
    lows = random.choices(range(4, 6), weights=[0.5, 0.5],  k = num1)
    
    for i in range(num1):
        fake_signals1.append(calc_lp(signal_array[i,0,:], highs[i], lows[i]))
    
    for i in range(num1, num1+num2):
        num = random.choice(range(5, 50))
        y = calc_lp(signal_array[i,0,:], 2, 1)
        z = y.copy()
        for k in range(num):
            step = random.choice(range(10, 100))
            z = z+np.roll(y, step)
        fake_signals2.append(normalize(z + calc_lp(signal_array[i,0,:], 7, 6)))
    fake_signals = fake_signals1 + fake_signals2   
    fake_signals = np.array(fake_signals).reshape(-1, 1, int(3*sRate))
    
    z_signals = np.random.standard_normal(size=(num_z, int(3*sRate)))
    y = np.concatenate([np.array([1,2,3,4],), np.arange(1, 150, 5)])
    x = np.random.choice(y, num_z, replace=True)
    scaled_signals = []
    for i in range(num_z):
        scaled_signals.append(z_signals[i]/x[i])

    scaled_signals = np.array(scaled_signals)
    z_signals = scaled_signals.reshape(-1, 1, int(3*sRate))
        
    fake_signals = np.vstack((fake_signals, z_signals))
    return fake_signals

# Main function
def pred_fine_tuned(sRate, anno, sig, name, path_input, path_model):
    
    print('Augmentation Start!')
#     anno, sig = process_prediction(name + '_signal_900.npy', name + '_fft_900.npy', 
#                                    'pretrained_noise_removal', 1, path_input, path_model)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 

    normal = sig[np.where(anno == 0)[0], :, :]
    z = []
    z.append(create_fake_noise(sRate, normal, num_z = 2000))
    z = np.vstack(z)
    sig = np.vstack((normal, z))
    anno = list(np.zeros(len(normal))) + list(np.ones(len(z)))
    ffts = np.absolute(fft(sig.squeeze()))
    ffts = np.asarray(ffts)
    ffts = np.expand_dims(ffts, 1)

    idx = list(range(sig.shape[0]))
    random.shuffle(idx)
    sig = sig[idx]

    ffts = ffts[idx]
    anno = np.array(anno)[idx]
    trn_sz = int(len(sig)*0.9)
    train_dataset,_,_= create_datasets((sig,ffts), anno,trn_sz, 0.2, seed=2021)
    datasets = create_datasets((sig,ffts), anno, trn_sz, 0.2, seed=2021)
    trn_dl, val_dl, tst_dl = create_loaders(datasets, 64)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(np.unique(anno, return_counts = True))
   
    print('Fine-tuning Start!')
    os.chdir(path_model)
    raw_feat = sig.shape[1]
    fft_feat = ffts.shape[1]
    
    sig, ffts, train_dataset, datasets = 0,0,0,0
    del sig, ffts, train_dataset, datasets
    gc.collect()
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 
    lr = 0.0001
    n_epochs = 500
    iterations_per_epoch = len(trn_dl)
    num_classes = 2
    best_acc = 0
    patience, trials = 10, 0
    base = 1
    step = 2
    loss_history = []
    acc_history = []
    
    if sRate == 300:
        model = Classifier_900(raw_feat, fft_feat, num_classes).to(device)
        state_dict = torch.load('pretrained_noise_removal_900', map_location='cuda:0')
    elif sRate == 250:
        model = Classifier_750(raw_feat, fft_feat, num_classes).to(device)
        state_dict = torch.load('pretrained_noise_removal_750', map_location='cuda:0')
    else:
        print('sRate must be 250 or 300!')

    
    model.load_state_dict(state_dict)
    print(sys.getsizeof(model))
    print(sys.getsizeof(state_dict))

    print('Load_model!')
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 
    for parameter in model.parameters():
        parameter.requires_grad = False

    layer_num = 0
    for param in model.raw.parameters():
        layer_num +=1
        if (layer_num > 19) | (layer_num < 2):
            param.requires_grad = True    

    n_features = 128
    model.out = nn.Linear(n_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations_per_epoch * 2, 
                                                           eta_min=lr/100)

    for epoch in range(1, n_epochs + 1):

        model.train()
        epoch_loss = 0
        for i, batch in enumerate(trn_dl):
            x_raw, x_fft, y_batch = [t.to(device) for t in batch]
            opt.zero_grad()
            out = model(x_raw, x_fft)
            loss = criterion(out, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
            scheduler.step()

        epoch_loss /= trn_sz
        loss_history.append(epoch_loss)

        model.eval()
        correct, total = 0, 0
        for batch in val_dl:
            x_raw, x_fft, y_batch = [t.to(device) for t in batch]
            out = model(x_raw, x_fft)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

        acc = correct / total
        acc_history.append(acc)

        if epoch % base == 0:
            print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
            base *= step

        if acc > best_acc:
            trials = 0
            best_acc = acc
            torch.save(model.state_dict(), 'fine')
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break

    print('Done!')
    
    trn_dl, val_dl, tst_dl = 0, 0, 0
    del trn_dl, val_dl, tst_dl
    gc.collect()
    print('Prediction Start!')
    
    os.chdir(path_input)
    test = np.load(name + '_signal_' + str(int(3*sRate)) + '.npy')
    test_fft = np.load(name + '_fft_' + str(int(3*sRate)) + '.npy')
    print(sys.getsizeof(test))
    print(sys.getsizeof(test_fft))
    target_te = np.zeros(len(test))

    sign_change_idx = np.where(count_sign_changes_array(test)/int(3*sRate) > 0.075)[0]

    tst_dl = create_testset((test, test_fft), target_te)
    tst_dl = create_loaders_test(tst_dl, bs=256)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     torch.set_num_threads(1)
    model.eval()

    # Prediction process

    pred = []
    probs = []
    for batch in tst_dl:
        x_raw, x_fft, y_batch = [t.to(device) for t in batch]
        out = model(x_raw, x_fft)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        pred.append(preds.tolist())

    pred = np.asarray(list(chain.from_iterable(pred)))
    flag = np.max(np.absolute(test), axis = 2).squeeze()
    out_amp_idx = np.where((flag < 0.075) | (flag > 2.0))[0]   
    
    pred[out_amp_idx] = 1
    pred[sign_change_idx] = 1

    return pred, test



class QRS_to_Noise:
    
    def __init__(self, name, sRate, num_threads, path_raw, path_input, path_model,
                 cut_wide=450, cut_height=0.000025, src_window1=70):
        os.chdir(path_raw)
        self.path_raw = path_raw
        self.path_input = path_input
        self.path_model = path_model
        self.name = name
        self.sRate = sRate
        self.num_threads = num_threads
        self.cut_wide = cut_wide
        self.cut_height = cut_height
        self.src_window1 = src_window1
        
    def wfdb_to_npy(self):
        os.chdir(self.path_raw)
        print('Processing wfdb file to numpy array')
        raw = self.name 
        
        if (self.name+'.npy' in os.listdir()) == False:
            raw_signal = wfdb.rdrecord(raw)
            np_signal = raw_signal.p_signal
            np.save(self.name + '.npy', np_signal)
 
    def save_dl_inputs(self, file_name):
        print('Saving deep learning inputs to directory')
        os.chdir(self.path_input)
        
        if self.sRate == 300:
            if ((self.name + '_signal_900.npy' in os.listdir()) == False) | ((self.name + '_fft_900.npy' in os.listdir()) == False):

                create_input_900(file_name, self.name, self.path_raw, self.path_input)
          
        elif self.sRate == 250:
            if ((self.name + '_signal_750.npy' in os.listdir()) == False) | ((self.name + '_fft_750.npy' in os.listdir()) == False):

                create_input_750(file_name, self.name, self.path_raw, self.path_input)
        else:
            print('sRate must be 250 or 300!')
            
    def noise_removal(self, option):
        if self.sRate == 300:
            self.a, self.b = process_prediction(self.sRate, self.name + '_signal_900.npy', self.name + '_fft_900.npy', 'pretrained_noise_removal_900', 1, self.path_input, self.path_model)
            gc.collect()
        elif self.sRate == 250:
            self.a, self.b = process_prediction(self.sRate, self.name + '_signal_750.npy', self.name + '_fft_750.npy', 
                                       'pretrained_noise_removal_750', 1, self.path_input, self.path_model)
        
#         self.pred, self.raw_signal = pred_fine_tuned(self.name, self.path_input, self.path_model)
        self.pred, self.raw_signal = pred_fine_tuned(self.sRate, self.a, self.b, self.name, self.path_input, self.path_model)
        del self.a, self.b
        gc.collect()
        
        self.pred = np.array(self.pred)
        
        if option != 'light':
            print('Running Auto-Encoder')
#             x = self.raw_signal
#             y = np.reshape(x ,len(x)*900)
#             y = calc_lp(y, 8, 2)
            self.pred2 = np.array(Running_AE(self.raw_signal, self.path_model)).squeeze()
            self.pred[np.where(self.pred2 == 1)] = 1
        self.clean = self.raw_signal[np.where(self.pred == 0)[0], 0, :]
        self.clean2 = self.clean.reshape(-1, int(3*self.sRate))
        self.clean = self.clean.reshape(self.clean.shape[0]*self.clean.shape[1])
        self.filtered = calc_lp(self.clean, 8, 1)
        self.noise = self.raw_signal[np.where(self.pred == 1)[0], 0, :]
        self.noise = self.noise.reshape(-1, int(3*self.sRate))
       
    def qrs_detection(self):
        print('R_peadk detection Start')
        self.anno = rpeak_final(self.clean, self.filtered, self.sRate, self.cut_wide,
                          self.cut_height, self.src_window1, self.num_threads, 'original')
        self.rr_pre = self.anno - np.roll(self.anno, 1)
        self.rr_pre[0] = self.sRate
        
        self.temp = pd.DataFrame({'pos': self.anno, 'target': 'N'})
        self.temp['rr_pre'] = self.temp['pos'] - np.roll(self.temp['pos'], 1)
        self.temp['rr_pre'][0] = self.sRate
        self.temp['rr_post'] = np.roll(self.temp['pos'], -1) - self.temp['pos'] 
        self.temp['rr_post'][len(self.temp)-1] = self.sRate
        self.temp['rr_ratio'] = self.temp['rr_pre'] / self.temp['rr_post']
    
    def afib_detection(self, num_workers_):
        print('Afib detection Start')
        self.pred_afib = A_FIB_Detection(np.array(self.temp.pos), self.sRate, self.path_model, num_workers_)
    
    def apcrun_detection(self):
        print('APC_Run detection Start')
        self.pred_arun = apcrun_detector(np.array(self.temp.pos))
        self.pred_arun = np.array(list(chain.from_iterable(self.pred_arun)))
    
    def create_output(self):
        print('Creating Output')
#         self.temp = self.temp[:len(self.pred_afib)]
#         self.temp['afib'] = self.pred_afib
#         self.temp['apc_run'] = self.pred_arun
#         self.temp.afib = self.temp.afib.astype(int)
#         self.temp.apc_run = self.temp.apc_run.astype(int)
        amt_add = get_sum_pre_noise(self.pred)
        bins = np.arange(0, self.anno[-1]+1, int(3*self.sRate))
        window_idx = np.digitize(self.anno, bins, right = True)
        add_matrix = pd.DataFrame({'idx': range(1, len(amt_add)+1),
                                  'amt': amt_add})
        rpeak_matrix = pd.DataFrame({'pos': self.anno,
                                    'idx': window_idx})
        rpeak_matrix = pd.merge(rpeak_matrix, add_matrix, left_on='idx', right_on='idx', how='left')
        rpeak_matrix['pos'] = rpeak_matrix['pos'] + rpeak_matrix['amt']
        rpeak_matrix['pos'] = pd.to_numeric(rpeak_matrix['pos'], downcast='integer')
        rpeak_matrix = rpeak_matrix.drop(labels=['idx', 'amt'], axis=1)
        rpeak_matrix['target'] = 'N'
        full_signal = self.raw_signal.reshape(self.raw_signal.squeeze().shape[0]*
                                              self.raw_signal.squeeze().shape[1])
        
        x = len(np.where(self.pred == 1)[0])/len(self.pred)
        summary = get_summary_anno(self.rr_pre, x*100)
        
        return rpeak_matrix, full_signal, summary, self.temp, self.filtered, self.clean2, self.noise
        