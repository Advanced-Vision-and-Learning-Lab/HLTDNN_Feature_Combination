#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:58:08 2024

@author: amir.m
"""

import torch
import torch.nn as nn
import lightning as L
from Utils.RBFHistogramPooling import HistogramLayer
from Prepare_Data import Prepare_DataLoaders
from Utils.Network_functions import initialize_model
from Utils.TDNN import TDNN

import torch.nn.functional as F
import numpy as np
import os
import argparse

from itertools import product

from Utils.Save_Results import save_results, get_file_location

from Demo_Parameters import Parameters
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import pdb

class LitModel(L.LightningModule):

    def __init__(self, HistogramLayer, Params, model_name, num_classes, in_channels,kernel_size,num_feature_maps,feat_map_size, numBins, dataset, dataset_dimension):
        super().__init__()
        kernel_size = Params['kernel_size'][model_name]
        in_channels = Params['in_channels'][model_name] 

        saved_bins = np.zeros((Params['num_epochs'] + 1,
                               numBins * int(num_feature_maps / (feat_map_size * numBins))))
        saved_widths = np.zeros((Params['num_epochs'] + 1,
                                 numBins * int(num_feature_maps / (feat_map_size * numBins))))


        histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                         kernel_size,
                                         num_bins=numBins, stride=Params['stride'],
                                         normalize_count=Params['normalize_count'],
                                         normalize_bins=Params['normalize_bins'])
    


        self.model_ft, input_size, self.feature_extraction_layer = initialize_model(model_name, num_classes,
                                                in_channels,
                                                num_feature_maps,
                                                feature_extract=Params['feature_extraction'],
                                                histogram=Params['histogram'],
                                                histogram_layer=histogram_layer,
                                                parallel=Params['parallel'],
                                                use_pretrained=Params['use_pretrained'],
                                                add_bn=Params['add_bn'],
                                                scale=Params['scale'],
                                                feat_map_size=feat_map_size,
                                                TDNN_feats=(Params['TDNN_feats'][dataset] * len(Params['feature'])),
                                                input_features = Params['feature'], 
                                                dataset_dimension = dataset_dimension)

        
        # Save the initial values for bins and widths of histogram layer
        # Set optimizer for model
        if (Params['histogram']):
            reduced_dim = int((num_feature_maps / feat_map_size) / (numBins))
            if (in_channels == reduced_dim):
                dim_reduced = False
                saved_bins[0, :] = self.model_ft.module.histogram_layer.centers.detach().cpu().numpy()
                saved_widths[0, :] = self.model_ft.module.histogram_layer.widths.reshape(
                    -1).detach().cpu().numpy()
            else:
                dim_reduced = True
                saved_bins[0, :] = self.model_ft.module.histogram_layer[
                    -1].centers.detach().cpu().numpy()
                saved_widths[0, :] = self.model_ft.module.histogram_layer[-1].widths.reshape(
                    -1).detach().cpu().numpy()
        else:
            saved_bins = None
            saved_widths = None
            dim_reduced = None
            
        self.save_hyperparameters()
            
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, z = batch
        # x = x.view(x.size(0), -1)
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        loss = F.cross_entropy(y_pred, y)
        self.log('val_loss', loss)
        return loss    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_ft.parameters(), lr=1e-3)
        return optimizer    
    
    


def main(Params):
    
    # Name of dataset
    dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name']
    
    # Number of classes in dataset
    num_classes = Params['num_classes'][dataset]
    
    # Number of runs and/or splits for dataset
    num_runs = Params['Splits'][dataset]
    
    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][model_name]
    
    feat_map_size = Params['feat_map_size']
    
    
    kernel_size = Params['kernel_size'][model_name]
    in_channels = Params['in_channels'][model_name] 
    
    
    print('Starting Experiments...')
    for split in range(0, num_runs-2):
        
        #Create dataloaders for model training
        dataloaders_dict, dataset_dimension = Prepare_DataLoaders(Params)
        
        #Get location to save results
        filename = get_file_location(Params,split)
        
        #Change filename to remov
        logger_name = '{}{}'.format(filename.split('Run')[0],'Summary/')
        logger = TensorBoardLogger(logger_name, name='', version ='Run {}'.format(split+1))
        checkpoint_callback = ModelCheckpoint(dirpath=filename,
                    monitor='val_loss',
                    filename='best_model')
        

        train_loader = dataloaders_dict['train']
        validation_loader = dataloaders_dict['val']
    

        model = LitModel(HistogramLayer, Params, model_name, num_classes, 
                         in_channels,kernel_size,num_feature_maps, feat_map_size, 
                         numBins, dataset, dataset_dimension)

        trainer = L.Trainer(accelerator='auto', devices='auto', max_epochs=5, 
                            logger=logger,
                            default_root_dir=os.path.join(os.getcwd(), 'lightning_logs'),
                            callbacks=[EarlyStopping(monitor="val_loss", mode="min",
                                                  patience=Params['patience'],verbose=False),
                                    checkpoint_callback],
                            enable_progress_bar=True)
        
        torch.set_float32_matmul_precision('medium')
        trainer.fit(model, train_loader, validation_loader)


        # TEST 
        
        # SAVE RESULTS
   

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/Lightning_Test/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='TDNN',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=0,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', nargs='+', default=['CQT', 'VQT', 'MFCC', 'STFT'], 
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type = str, default = 'Adagrad',
                       help = 'Select optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to train each model for (default: 50)')
    args = parser.parse_args()
    return args
        


if __name__ == "__main__":
    args = parse_args()
    
    #Create feature list for all 64 combinations
    feature_list = ['Mel_Spectrogram', 'CQT', 'VQT', 'MFCC', 'STFT', 'GFCC']
    
    #Generate binary combinations
    settings = list(product((True, False), repeat=len(feature_list)))
    
    #Remove last feature setting
    settings.pop(-1)
    
    setting_count = 1
    
    for setting in settings:
        
        #Take feature setting and select features
        temp_features = []
        count = 0
        for current_feature in setting:
            if current_feature:
                temp_features.append(feature_list[count])
            count += 1

        setattr(args, 'audio_feature', temp_features)
        params = Parameters(args)
        main(params)
        print('Finished setting {} of {}'.format(setting_count,len(settings)))
        setting_count += 1      
        if setting_count == 2:
            break
        
