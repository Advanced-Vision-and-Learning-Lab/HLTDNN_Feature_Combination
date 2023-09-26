# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 19:15:42 2023
Code modified from: https://github.com/lucascesarfd/underwater_snd/blob/master/nauta/one_stage/dataset.py
@author: jpeeples
"""

import torchaudio
import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Datasets.Get_Audio_Features import Get_Audio_Features
import pdb

class DeepShipSegments(Dataset):
    def __init__(self, parent_folder, train_split=.7,val_test_split=.5,
                 partition='train', random_seed= 42, shuffle = False, transform=None, 
                 target_transform=None, features=None):
        self.parent_folder = parent_folder
        self.folder_lists = {
            'train': [],
            'test': [],
            'val': []
        }
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.partition = partition
        self.transform = transform
        self.shuffle = shuffle
        self.target_transform = target_transform
        self.random_seed = random_seed
        self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        # self.feature_extraction_layer = feature_extraction_layer
        self.features = features

        # Loop over each label and subfolder
        for label in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
            label_path = os.path.join(parent_folder, label)
            subfolders = os.listdir(label_path)
            
            # Split subfolders into training, testing, and validation sets
            subfolders_train, subfolders_test_val = train_test_split(subfolders, 
                                                                     train_size=train_split, 
                                                                     shuffle=self.shuffle, 
                                                                     random_state=self.random_seed)
            subfolders_test, subfolders_val = train_test_split(subfolders_test_val, 
                                                               train_size=self.val_test_split, 
                                                               shuffle=self.shuffle, 
                                                               random_state=self.random_seed)

            # Add subfolders to appropriate folder list
            for subfolder in subfolders_train:
                subfolder_path = os.path.join(label_path, subfolder)
                self.folder_lists['train'].append((subfolder_path, self.class_mapping[label]))

            for subfolder in subfolders_test:
                subfolder_path = os.path.join(label_path, subfolder)
                self.folder_lists['test'].append((subfolder_path, self.class_mapping[label]))

            for subfolder in subfolders_val:
                subfolder_path = os.path.join(label_path, subfolder)
                self.folder_lists['val'].append((subfolder_path, self.class_mapping[label]))

        self.segment_lists = {
            'train': [],
            'test': [],
            'val': []
        }

        # Loop over each folder list and add corresponding files to file list
        for split in ['train', 'test', 'val']:
            for folder in self.folder_lists[split]:
                for root, dirs, files in os.walk(folder[0]):
                    for file in files:
                        if file.endswith('.wav'):
                            file_path = os.path.join(root, file)
                            label = folder[1]
                            self.segment_lists[split].append((file_path, label))

    def __len__(self):
        return len(self.segment_lists[self.partition])

    def __getitem__(self, idx):
        file_path, label = self.segment_lists[self.partition][idx]
        signal, sr = torchaudio.load(file_path, normalize = True)

        # signal = signal.to(device)  # Move data to the device if needed
        # move signal to device
        # define device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # signal = signal.to(device)
        # if self.feature_extraction_layer is None:
        #     raise RuntimeError("Feature extraction layer is not defined.")

        # extracted_features = self.feature_extraction_layer(signal)
        # extracted_features = model.feature_extraction_layer(signal)
        # pdb.set_trace()
        combined_features = []
        if self.features:
            for feature in self.features:
                data_transforms = Get_Audio_Features(feature)
                # print('data_transforms: ', data_transforms)
                if self.partition == 'train':
                    temp_transform = data_transforms['train']
                else:
                    temp_transform = data_transforms['test']
                
                
                transformed_signal = temp_transform(signal)
                # print('transfrom shape: ', transformed_signal.shape)
                
                combined_features.append(transformed_signal)
        
        # resized_features = []
        # for feature in combined_features:
        #     print('feature.shape: ', feature.shape)
        #     feat = feature.view(feature.size(0), feature.size(1), -1)
        #     print('resized feat shape: ', feat.shape)
        #     resized_features.append(feat)
        combined_features = torch.cat(combined_features, dim=0)
        # print('combined_features shape: ', combined_features.shape)
        label = torch.tensor(label)
        # pdb.set_trace()
        if self.target_transform:
            label = self.target_transform(label)

        return combined_features, label, idx