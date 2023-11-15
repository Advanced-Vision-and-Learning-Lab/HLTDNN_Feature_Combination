import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import transforms
from nnAudio import features
import pdb

import torch
from Datasets.Adaptive_Pad_Layer import Adaptive_Pad_Layer


class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_features, dataset_dimension, sample_rate=16000, window_length=250, hop_length=64, RGB=False, pretrained=False):
        super(Feature_Extraction_Layer, self).__init__()
        
        #Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        self.test_tensor_height = 1
        self.test_tensor_width = 48000
        
        if RGB:
            num_channels = 3
        else:
            num_channels = 1

        self.input_features = input_features
        # Create a test tensor with shape of dataset to generate dimensions of feature tensors 
        self.test_tensor = torch.rand((dataset_dimension[-2], dataset_dimension[-1]))

        #Return Mel Spectrogram that is 48 x 48
        self.Mel_Spectrogram = nn.Sequential(features.mel.MelSpectrogram(sample_rate,n_mels=44,win_length=int(window_length*sample_rate),
                                            hop_length=int(hop_length*sample_rate),
                                            n_fft=int(window_length*sample_rate), verbose=False), nn.ZeroPad2d((1,4,0,4)))
        
    
        #Return MFCC that is 16 x 48
        self.MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16, 
                                        n_fft=int(window_length*sample_rate), 
                                                win_length=int(window_length*sample_rate), 
                                                hop_length=int(hop_length*sample_rate),
                                                n_mels=48, center=False, verbose=False), nn.ZeroPad2d((1,0,4,0)))

        #Return STFT that is 48 x 48
        self.STFT = nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                        hop_length=int(hop_length*sample_rate),
                                        win_length=int(window_length*sample_rate), 
                                        output_format='Magnitude',
                                        freq_bins=48,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return GFCC that is 64 x 48
        self.GFCC = nn.Sequential(features.Gammatonegram(sr=sample_rate,
                                                hop_length=int(hop_length*sample_rate),
                                                n_fft=int(window_length*sample_rate),
                                                verbose=False,n_bins=64), nn.ZeroPad2d((1,0,0,0)))
        

        #Return CQT that is 64 x 48
        self.CQT = nn.Sequential(features.CQT(sr=sample_rate, n_bins=64, 
                                        hop_length=int(hop_length*sample_rate),
                                        verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return VQT that is 64 x 48
        self.VQT = nn.Sequential(features.VQT(sr=sample_rate,hop_length=int(hop_length*sample_rate),
                                        n_bins=64,earlydownsample=False,verbose=False), nn.ZeroPad2d((1,0,0,0)))

        self.features = {'Mel_Spectrogram': self.Mel_Spectrogram, 'MFCC': self.MFCC, 'STFT': self.STFT, 'GFCC': self.GFCC, 'CQT': self.CQT, 'VQT': self.VQT}
        self.max_height, self.max_width = self.get_max_tensor_dimension()
        self.adaptive_pad_layer = Adaptive_Pad_Layer(self.max_height, self.max_width)

    def get_max_tensor_dimension(self):
        max_height = 0
        max_width = 0
        for feature in self.features.values():
            result = feature[0](self.test_tensor)
            height, width = result.size(-2), result.size(-1)
            max_height = max(max_height, height)
            max_width = max(max_width, width)
        return max_height, max_width


    def forward(self, x):
        transformed_features = []
        pdb.set_trace()
        for feature in self.input_features:
            feature_tensor = self.features[feature](x)
            padded_tensor = self.adaptive_pad_layer(feature_tensor) 
            transformed_features.append(padded_tensor)

        combined_features = torch.stack(transformed_features, dim=1)
        return combined_features
