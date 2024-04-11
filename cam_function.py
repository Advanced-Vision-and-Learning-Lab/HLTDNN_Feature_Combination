#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:15:47 2024

@author: amir.m
"""


# cam_function.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import re

def generate_CAM(model, feature_extraction_layer, dataloaders_dict, device, sub_dir, device_loc, Params):

    print('Loading model...')
    model.load_state_dict(torch.load(sub_dir + 'Best_Weights.pt', map_location=device_loc))
    model = model.to(device)
    feature_extraction_layer = feature_extraction_layer.to(device)

    dataloader = dataloaders_dict['train']
    
    model.eval()
    feature_extraction_layer.eval()
    

    count_label_0 = 0
    target_sample_number = 5
    found = False

    for batch in dataloader:
        signals, labels, _ = batch
        signals = signals.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            for signal, label in zip(signals, labels):
                if label.item() == 0:
                    count_label_0 += 1
                    
                    if count_label_0 == target_sample_number:
                        print('Found the 5th sample with label 0')
                        sample_signal = signal
                        sample_label = label
                        found = True
                        break
            if found:
                break

    input_tensor = feature_extraction_layer(sample_signal.unsqueeze(0))
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class_index = torch.max(probabilities, dim=1)
        target_class_index = predicted_class_index.item()
    print(f"Target Class Index: {target_class_index}")
    
    
    predicted_class = predicted_class_index.cpu().numpy()[0]
    actual_class = sample_label.cpu().item()  
    if predicted_class == actual_class:
        print("The sample was correctly classified by the model.")
    else:
        print("The sample was misclassified by the model.")
        
    
    target_layers = [model.module.backbone.conv5, model.module.histogram_layer.bin_widths_conv]
    targets = [ClassifierOutputTarget(target_class_index)]
    cam = FullGrad(model=model.module, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    


    feature_y_axis_labels = {
        'Mel_Spectrogram': 'Frequency (Hz)',
        'CQT': 'Frequency (Hz)',
        'VQT': 'Frequency (Hz)',
        'MFCC': 'Coefficients',
        'STFT': 'Frequency (Hz)',
        'GFCC': 'Coefficients'
    }
    
    x_axis_label = 'Time (s)'
    

    match = re.search(r'Run_(\d+)', sub_dir)
    run_number = match.group(1) if match else None


    for i, feature in enumerate(Params['feature']):
        # Check if the feature is in the feature_y_axis_labels dictionary
        if feature in feature_y_axis_labels:
            # Fetch the original image data
            original_img = input_tensor[0][i:i+1].detach().cpu().numpy()
            original_img = np.transpose(original_img, (1, 2, 0))
            
            # Normalize the input channel for CAM visualization
            rgb_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            rgb_img = rgb_img.astype(np.float32)
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Save original image
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(original_img, aspect='auto')
            ax.set_xlabel(x_axis_label, fontsize=14, fontweight='bold')
            ax.set_ylabel(feature_y_axis_labels[feature], fontsize=14, fontweight='bold')
            ax.axis('on')
            cbar = fig.colorbar(im, ax=ax, fraction=0.12, pad=0.04)
            cbar.ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'CAM_Figures/Original_{feature}_pclass_{target_class_index}_aclass_{actual_class}_run{run_number}.png', dpi=150)
            plt.close(fig)
            
            # Save CAM overlay image
            fig, ax = plt.subplots(figsize=(4, 4))
            cam_image = ax.imshow(visualization, aspect='auto')
            ax.set_xlabel(x_axis_label, fontsize=14, fontweight='bold')
            ax.set_ylabel(feature_y_axis_labels[feature], fontsize=14, fontweight='bold')
            ax.axis('on')
            vmin, vmax = grayscale_cam.min(), grayscale_cam.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
            sm.set_array([])
            cbar_cam = fig.colorbar(sm, ax=ax, fraction=0.12, pad=0.04)
            cbar_cam.ax.set_ylabel('Activation', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'CAM_Figures/CAM_Overlay_{feature}_pclass_{target_class_index}_aclass_{actual_class}.png', dpi=150)
            plt.close(fig)
        