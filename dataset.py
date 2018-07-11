__author__ = 'Richard Diehl Martinez' 

'''
Creates the basic dataset that is fed into the sampler for the PyTorch 
Model. 
'''

import numpy as np
import random
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
from enum import Enum 

class MoleType(Enum): 
    benign = 0 
    malignant = 1 
    indeterminate = 2 
 
class ImageDataset(): 
    def __init__(self, path = 'cleaned_data.pkl', shuffle = True):
                       
        """
        Args:
            path (string): Path to the .pkl file that contains all the 
            data 
        """
        self.shuffle = shuffle 
        
        self.path = path
        self.dataset = self.extract_info() # list of matrices
        self.lookup = MoleType
        self.trans = transforms.ToTensor()
        self.resize = transforms.Resize((144,192))

    def __len__(self):
        '''
        Returns:
            total: sums over the length of the individual data sets
        '''
        return len(self.dataset)
        
    def __getitem__(self,idx):
        img_path, label = self.dataset[idx] 
        
        # Applying Transformation to image
        img = self.trans(self.resize(Image.open(img_path)))

        try: 
            label = self.lookup[label].value
        except:
            print('error: could not find label', label)
            exit()
        return img, label 
        
    def extract_info(self): 
        full_data = pickle.load(open(self.path,'rb'))
        if self.shuffle: 
            random.shuffle(full_data)
        return full_data

def main(): 
    print('--- Running Test for ImageDataset class ---')
    dataset = ImageDataset()
    print('Length of Dataset', len(dataset))  
    index = 300
    print('Returning Item at index: ', index)
    print(dataset[index])
    
    print('Printing out shapes of next 10 images')
    print('Image shape: ', dataset[index][0].shape) 
    for i in range(10):
        print('Image shape: ', dataset[index+i][0].shape) 
        print('Image labels: ', dataset[index+i][1]) 
 
if __name__ == '__main__': 
   main()