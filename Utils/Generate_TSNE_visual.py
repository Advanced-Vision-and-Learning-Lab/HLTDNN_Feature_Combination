# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:31:02 2020

@author: jpeeples
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import numpy as np
import torch
from matplotlib import offsetbox
from Utils.Compute_FDR import Compute_Fisher_Score
from Datasets.Get_Audio_Features import Get_Audio_Features
import pdb

def plot_components(data, proj, images=None, ax=None,
                    thumb_frac=0.05, cmap='copper'):
    ax = ax or plt.gca()
    
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i],zoom=.2, cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)
            
def Generate_TSNE_visual(dataloaders_dict,model,sub_dir,device,class_names, input_features=None,
                         histogram=True,Separate_TSNE=False, feature_layer=None):

      # Turn interactive plotting off, don't show plots
        plt.ioff()
      #TSNE visual of (all) data
        #Get labels and outputs
        for phase in ['train', 'val', 'test']:
            GT_val = np.array(0)
            indices_train = np.array(0)
            model.eval()
            model.to(device)
            feature_layer.eval()
            feature_layer.to(device)
            features_extracted = []
            saved_imgs = []
            for idx, (inputs, classes,index)  in enumerate(dataloaders_dict[phase]):
                # features = Get_Audio_Features(input_features, inputs)
                images = inputs.to(device)
                labels = classes.to(device, torch.long)
                indices  = index.to(device).cpu().numpy()
                
                GT_val = np.concatenate((GT_val, labels.cpu().numpy()),axis = None)
                indices_train = np.concatenate((indices_train,indices),axis = None)
                
                images = feature_layer(images) 
                features = model(images)
                    
                features = torch.flatten(features, start_dim=1)
                
                features = features.cpu().detach().numpy()
                
                features_extracted.append(features)
                saved_imgs.append(images.cpu().permute(0,2,3,1).numpy())
                break
            
      
            features_extracted = np.concatenate(features_extracted,axis=0)
            saved_imgs = np.concatenate(saved_imgs,axis=0)
            
            #Compute FDR scores
            GT_val = GT_val[1:]
            indices_train = indices_train[1:]
            FDR_scores, log_FDR_scores = Compute_Fisher_Score(features_extracted,GT_val)
            np.savetxt((sub_dir+'{}_FDR.txt'.format(phase)),FDR_scores,fmt='%.2E')
            np.savetxt((sub_dir+'{}_log_FDR.txt'.format(phase)),log_FDR_scores,fmt='%.2f')
            features_embedded = TSNE(n_components=2,verbose=1,init='random',random_state=42).fit_transform(features_extracted)
        
            fig6, ax6 = plt.subplots()
            colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
            for texture in range (0, len(class_names)):
                x = features_embedded[[np.where(GT_val==texture)],0]
                y = features_embedded[[np.where(GT_val==texture)],1]
                
                ax6.scatter(x, y, color = colors[texture,:],label=class_names[texture])
             
            plt.title('TSNE Visualization of {} Data Features'.format(phase.capitalize()))
            
            box = ax6.get_position()
            ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            ax6.legend(loc='upper center',bbox_to_anchor=(.5,-.05),fancybox=True,ncol=4)
            plt.axis('off')
            
            fig6.savefig((sub_dir + 'TSNE_Visual_{}_Data.png'.format(phase.capitalize())), dpi=fig6.dpi)
            plt.close()
            
            #Plot tSNE with images
            fig9, ax9 = plt.subplots()
            # plot_components(features_extracted,features_embedded,thumb_frac=0.1,
            #                 images=saved_imgs,cmap=None)
            plt.grid('off')
            plt.axis('off')
            
            fig9.savefig((sub_dir + 'TSNE_Visual_{}_Data_Images.png'.format(phase.capitalize())),dpi=fig9.dpi)
            plt.close()
    
        # del dataloaders_dict,features_embedded
        torch.cuda.empty_cache()
        
        return FDR_scores, log_FDR_scores