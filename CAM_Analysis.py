#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:48:46 2024

@author: amir.m
"""

import torch

import argparse
import numpy as np
import torch.nn as nn

## Local external libraries
from Utils.Network_functions import initialize_model
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Save_Results import save_results
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
import matplotlib.pyplot as plt
import pdb

import PIL.Image
    
def main(Params):
    
    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name']
    
    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][model_name]
    
    # Local area of feature map after histogram layer
    feat_map_size = Params['feat_map_size']
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", torch.cuda.device_count(), "GPUs!")
    

    histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                     Params['kernel_size'][model_name],
                                     num_bins=numBins, stride=Params['stride'],
                                     normalize_count=Params['normalize_count'],
                                     normalize_bins=Params['normalize_bins'])

    # Create training and validation dataloaders
    dataloaders_dict, dataset_dimension = Prepare_DataLoaders(Params)

    # Initialize the histogram model for this run
    model_ft, input_size, feature_extraction_layer = initialize_model(model_name, num_classes,
                                            Params['in_channels'][model_name],
                                            num_feature_maps,
                                            feature_extract=Params['feature_extraction'],
                                            histogram=Params['histogram'],
                                            histogram_layer=histogram_layer,
                                            parallel=Params['parallel'],
                                            use_pretrained=Params['use_pretrained'],
                                            add_bn=Params['add_bn'],
                                            scale=Params['scale'],
                                            feat_map_size=feat_map_size,
                                            TDNN_feats=(Params['TDNN_feats'][Dataset] * len(Params['feature'])),
                                            input_features = Params['feature'], 
                                            dataset_dimension = dataset_dimension)    
    
    
    
    
    # Send the model to GPU if available, use multiple if available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)
    feature_extraction_layer = feature_extraction_layer.to(device)
    
    # Print number of trainable parameters
    num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print("Initializing Datasets and Dataloaders...")
    print("Number of params: ", num_params)
      

    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image    


    model_ft.load_state_dict(torch.load('Best_Weights.pt'))
    
    dataloader = dataloaders_dict['train']


    model_ft.eval()
    feature_extraction_layer.eval()
       

    # dataiter = iter(dataloader)
    # images, labels = next(dataiter)[:2]
    

    i=0
    for batch in dataloader:
        if i==3:
            first_batch = batch
            break
        i+=1
    
    
    sample_signal = first_batch[0][2] 
    sample_label = first_batch[1][2]   
    
    sample_signal = sample_signal.to(device)
    sample_label = sample_label.to(device)
    
 
        
    
    input_tensor = feature_extraction_layer(sample_signal.unsqueeze(0))  # Shape: [1, 4, H, W]
    # Forward pass 
    logits = model_ft(input_tensor)  
    # Convert logits to probabilities for multi-class classification
    probabilities = torch.softmax(logits, dim=1)
    # Determine predicted class
    _, predicted_class = torch.max(probabilities, 1)
    # Convert predicted_class and sample_label to CPU and extract their scalar values if they are tensors
    predicted_class = predicted_class.cpu().numpy()[0]
    actual_class = sample_label.cpu().item()  
    # Compare predicted class to actual class
    if predicted_class == actual_class:
        print("The sample was correctly classified by the model.")
    else:
        print("The sample was misclassified by the model.")
        
    
    with torch.no_grad():
        outputs = model_ft(input_tensor)
        # Assuming outputs are logits; apply softmax to convert to probabilities
        probabilities = torch.softmax(outputs, dim=1)
        # Get the predicted class index with the highest probability
        _, predicted_class_index = torch.max(probabilities, dim=1)
        target_class_index = predicted_class_index.item()

    print(f"Target Class Index: {target_class_index}")

    
    target_layers = [model_ft.module.backbone.conv5, model_ft.module.histogram_layer.bin_widths_conv]
    #target_layers = [model_ft.module.histogram_layer.bin_centers_conv] 
    #target_layers = [model_ft.module.backbone.conv5] 

    # Note: input_tensor can be a batch tensor with several images!

    # Define the target for which you want to generate the CAM
    targets = [ClassifierOutputTarget(target_class_index)]
    
    # Initialize Grad-CAM
    cam = FullGrad(model=model_ft.module, target_layers=target_layers)

    # Generate the CAM for the given input
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]  # Assuming a single image in the batch
          
    #pdb.set_trace()
    for i in range(4):
        # Preparing the image for overlaying CAM
        rgb_img = input_tensor[0][i:i+1].detach().cpu().numpy()  
        rgb_img = np.transpose(rgb_img, (1, 2, 0))  # Change from CxHxW to HxWxC
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # Normalize to [0, 1] for displaying
        
        # Overlay CAM on the image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # plt.figure(figsize=(8, 6))
        # plt.imshow(visualization)
        # plt.axis('off')
        # plt.savefig(f'CAM_Figures/CAM_visualization_{i}.png', dpi=500)  
        # plt.close()
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  
        axs[0].imshow(rgb_img)
        axs[0].set_title('Original Spectrogram')
        axs[0].axis('off')  
        
        axs[1].imshow(visualization)
        axs[1].set_title('CAM Overlay')
        axs[1].axis('off')  
        
        plt.savefig(f'CAM_Figures/combined_visualization_ch{i}_class{target_class_index}.png', dpi=500)
        plt.close(fig)  

    # #FullGrad
    # # Initialize EigenCAM with the selected target layer
    # cam = FullGrad(model=model_ft.module, target_layers=target_layers)

    # # Define the target for which you want to generate the CAM
    # targets = [ClassifierOutputTarget(target_class_index)]    
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0, :] 
    
    # # Normalize and convert CAM to correct format
    # grayscale_cam = np.maximum(grayscale_cam, 0)  
    # grayscale_cam = grayscale_cam - np.min(grayscale_cam)
    # grayscale_cam = grayscale_cam / np.max(grayscale_cam)
    # grayscale_cam = np.float32(grayscale_cam)
    
    # num_channels = input_tensor.shape[1]  # Number of channels in the input_tensor
    
    # for channel_index in range(num_channels):
    #     # Extract and normalize the channel image
    #     channel_img = input_tensor[0, channel_index].cpu().detach().numpy()
    #     channel_img_normalized = (channel_img - np.min(channel_img)) / (np.max(channel_img) - np.min(channel_img))
        
    #     # Convert to RGB format by stacking
    #     channel_img_rgb = np.stack((channel_img_normalized,) * 3, axis=-1)
        
    #     # Generate CAM visualization
    #     visualization = show_cam_on_image(channel_img_rgb, grayscale_cam, use_rgb=True)
        
    #     # Combine the original spectrogram channel and the CAM visualization side by side
    #     combined_image = np.hstack((channel_img_rgb, visualization))
        
    #     # Convert combined image to uint8
    #     combined_image_uint8 = np.uint8(255 * combined_image)
    
    #     plt.figure(figsize=(8, 6))  
    #     plt.imshow(combined_image_uint8, aspect='auto', origin='lower', cmap='viridis')  
    #     plt.colorbar(label='Magnitude')
    #     plt.title(f'Channel {channel_index}: Original Spectrogram and CAM Overlay')
    #     plt.axis('off') 
    #     plt.savefig(f'CAM_Figures/{channel_index}_virdis', dpi=500)
            

    #     plt.figure(figsize=(12, 6)) 
    #     plt.subplot(1, 2, 1)  
    #     plt.imshow(channel_img_normalized, aspect='auto', origin='lower', cmap='viridis')
    #     plt.colorbar(label='Magnitude')
    #     plt.title(f'Channel {channel_index} Spectrogram')
    #     plt.axis('off')  
    #     plt.subplot(1, 2, 2) 
    #     plt.imshow(visualization, aspect='auto', origin='lower')
    #     plt.title(f'Channel {channel_index} CAM Overlay')
    #     plt.axis('off')  # Hide axes for better visualization
    #     plt.savefig(f'CAM_Figures/Channel_{channel_index}__virdis_combined.png', dpi=500)
    #     plt.close()
  
    
  


def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='TDNN',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=True, action=argparse.BooleanOptionalAction,
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
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', nargs='+', default=['VQT', 'MFCC', 'STFT', 'GFCC'],
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type = str, default = 'Adagrad',
                       help = 'Select optimizer')
    parser.add_argument('--patience', type=int, default=15,
                        help='Number of epochs to train each model for (default: 50)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)

