#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:26:34 2023
Generate audio feature transforms
@author: jarin.ritu
"""
from torchvision import transforms
import torchaudio.transforms as T
from nnAudio import features
import librosa
import numpy as np
import pdb
import torch


def vqt_lib(x):
    x_numpy = x.numpy().astype(np.float32)
    signal = librosa.vqt(x_numpy, sr=16000,hop_length=int((64/1000)*16000),
                                        n_bins=64)
    signal = torch.from_numpy(signal)
    return signal

def Get_Audio_Features(feature, sample_rate=16000, window_length=250, 
                       hop_length=64, RGB=False, pretrained=False, device="cpu"):
    
    #Convert window and hop length to ms
    window_length /= 1000
    hop_length /= 1000
    
    if RGB:
        num_channels = 3
    else:
        num_channels = 1
    
    #Based on desired feature, return transformation
    if feature == 'Mel_Spectrogram':
        #Return Mel Spectrogram that is 48 x 48
        
        signal_transform = features.mel.MelSpectrogram(sample_rate,n_mels=40,win_length=int(window_length*sample_rate),
                                            hop_length=int(hop_length*sample_rate),
                                            n_fft=int(window_length*sample_rate))

        #Amount to pad feature
        padding = (1,4,0,4)
        
       
    elif feature == 'MFCC':
        #Return MFCC that is 16 x 48
        signal_transform = features.mel.MFCC(sr=sample_rate, n_mfcc=16, 
                                        n_fft=int(window_length*sample_rate), 
                                                   win_length=int(window_length*sample_rate), 
                                                   hop_length=int(hop_length*sample_rate),
                                                   n_mels=48, center=False)
        padding = (1,0,0,0)
        

    elif feature == 'STFT': 
    #Return STFT that is 48 x 48
        signal_transform = features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                         hop_length=int(hop_length*sample_rate),
                                         win_length=int(window_length*sample_rate), 
                                         output_format='Magnitude',
                                         freq_bins=48,verbose=False)
        padding = (1,0,0,0)
        
    elif feature == 'GFCC':
        #Return GFCC that is 64 x 48
        signal_transform = features.Gammatonegram(sr=sample_rate,
                                                  hop_length=int(hop_length*sample_rate),
                                                  n_fft=int(window_length*sample_rate),
                                                  verbose=False,n_bins=64)
        padding = (1,0,0,0)
        

    elif feature == 'CQT':
        #Return CQT that is 64 x 48
        signal_transform = features.CQT(sr=sample_rate, n_bins=64, 
                                        hop_length=int(hop_length*sample_rate),
                                        verbose=False)
        padding = (1,0,0,0)
        
    elif feature == 'VQT':
        #Return VQT that is 64 x 48

        signal_transform = features.VQT(sr=sample_rate,hop_length=int(hop_length*sample_rate),
                                        n_bins=64,earlydownsample=False,verbose=False)
        padding = (1,0,0,0)

    else:
        raise RuntimeError('{} not implemented'.format(feature))
            
    signal_transform = signal_transform.to(device)


    train_transforms = transforms.Compose([
        signal_transform,
        transforms.Pad(padding),
        # transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
    ])


    test_transforms = transforms.Compose([
        signal_transform,
        transforms.Pad(padding),
        # transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
    ])

    #If pretrained, add ImageNet normalization
    if (pretrained and RGB):
        norm_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transforms = transforms.Compose(train_transforms,norm_transform)
        test_transforms = transforms.Compose(test_transforms,norm_transform)
    
    return { 'train':train_transforms, 'test': test_transforms}