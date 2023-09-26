import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import transforms
from nnAudio import features

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_features, sample_rate=16000, window_length=250, hop_length=64, RGB=False, pretrained=False):
        super(Feature_Extraction_Layer, self).__init__()
        
        self.train_feature_transforms = []
        self.test_feature_transforms = []

        #Convert window and hop length to ms
        window_length /= 10000
        hop_length /= 10000

        if RGB:
            num_channels = 3
        else:
            num_channels = 1
        for feature in input_features:
            if feature == 'Mel_Spectrogram':
                #Return Mel Spectrogram that is 48 x 48
        
                signal_transform = T.MelSpectrogram(sample_rate,n_mels=40,win_length=int(window_length*sample_rate),
                                                    hop_length=int(hop_length*sample_rate),
                                                    n_fft=int(window_length*sample_rate))
                
                train_transforms = transforms.Compose([
                    signal_transform,
                    transforms.Pad((1,4,0,4)),
                    #transforms.Lambda(lambda x: x.repeat(1, num_channels,1,1)),
                ])
                
                test_transforms = transforms.Compose([
                        signal_transform,
                        transforms.Pad((1,4,0,4)),
                        #transforms.Lambda(lambda x: x.repeat(1, num_channels,1,1)),
                    ])
            elif feature == 'MFCC':
                #Return MFCC that is 48 x 48
                signal_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=48, 
                                                melkwargs={"n_fft": int(window_length*sample_rate), 
                                                        "win_length": int(window_length*sample_rate), 
                                                        "hop_length": int(hop_length*sample_rate),
                                                        "n_mels": 48, "center": False})
                
                train_transforms = transforms.Compose([
                    signal_transform,
                    transforms.Pad((1,0,4,0)),
                    #transforms.Lambda(lambda x: x.repeat(1, num_channels,1,1)),
            
                ])
                test_transforms = transforms.Compose([
                        signal_transform,
                        transforms.Pad((1,0,4,0)),
                        #transforms.Lambda(lambda x: x.repeat(1, num_channels,1,1)),
                    ])
            elif feature == 'STFT':
                #Return STFT that is 48 x 48
                signal_transform = features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                                hop_length=int(hop_length*sample_rate),
                                                win_length=int(window_length*sample_rate), 
                                                output_format='Magnitude',
                                                freq_bins=48,verbose=False)
                    
                train_transforms = transforms.Compose([
                    signal_transform,
                    transforms.Pad((1,0,0,0)),
                    #transforms.Lambda(lambda x: x.repeat(1, num_channels,1,1)),
                ])

                
                test_transforms = transforms.Compose([
                        signal_transform,
                        transforms.Pad((1,0,0,0)),
                        #transforms.Lambda(lambda x: x.repeat(1, num_channels,1,1)),
                    ])

            elif feature == 'GFCC':
                #Return GFCC that is 48 x 48
                signal_transform = features.Gammatonegram(sr=sample_rate,
                                                        hop_length=int(hop_length*sample_rate),
                                                        n_fft=int(window_length*sample_rate),
                                                        verbose=False,n_bins=48)
                
                train_transforms = transforms.Compose([
                    signal_transform,
                    transforms.Pad((1,0,0,0)),
                    #transforms.Lambda(lambda x: x.repeat(num_channels,1,1))
                ])

                test_transforms = transforms.Compose([
                        signal_transform,
                        transforms.Pad((1,0,0,0)),
                        #transforms.Lambda(lambda x: x.repeat(num_channels,1,1))
                    ])

            elif feature == 'CQT':
                #Return CQT that is 48 x 48
                signal_transform = features.CQT(sr=sample_rate, n_bins=48, 
                                                hop_length=int(hop_length*sample_rate),
                                                verbose=False)
                
                train_transforms = transforms.Compose([
                    signal_transform,
                    transforms.Pad((1,0,0,0)),
                    #transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                ])

                test_transforms = transforms.Compose([
                        signal_transform,
                        transforms.Pad((1,0,0,0)),
                        #transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                    ])
                
            elif feature == 'VQT':
                #Return VQT that is 48 x 48
                signal_transform = features.VQT(sr=sample_rate,hop_length=int(hop_length*sample_rate),
                                                n_bins=48,earlydownsample=False,verbose=False)

                train_transforms = transforms.Compose([
                    signal_transform,
                    transforms.Pad((1,0,0,0)),
                    #transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                ])

            
                test_transforms = transforms.Compose([
                        signal_transform,
                        transforms.Pad((1,0,0,0)),
                        #transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                    ])
            # Add other feature transforms (GFCC, CQT, VQT, etc.) similarly
            
            self.train_feature_transforms.append(train_transforms)
            self.test_feature_transforms.append(test_transforms)
        
        self.num_input_features = len(input_features)
        
    def forward(self, x):
        transformed_features = []
        original_device = x.device
        # move to deivce 0 if available
        print("x.device: ", x.device)
        x = x.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        for i in range(self.num_input_features):

            # move transforms in list to device
            transforms = [
                t.to(x.device) if isinstance(t, torch.nn.Module) else t
                for t in self.train_feature_transforms[i].transforms
            ]
            transformed_feature = self.train_feature_transforms[i](x)
            transformed_features.append(transformed_feature)

        combined_features = torch.cat([
            feature.unsqueeze(1) for feature in transformed_features
        ], dim=1)
        # move back to original device
        combined_features = combined_features.to(original_device)
        return combined_features
