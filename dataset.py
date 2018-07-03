__author__ = 'Richard Diehl Martinez' 

'''
Creates the basic dataset that is fed into the sampler for the PyTorch 
Model. 
'''

import numpy as np
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torch
from enum import Enum 

class MoleType(Enum): 
    benign = 0 
    malignant = 1 
 
 
class ImageDataset(): 
    def __init__(self, path = 'cleaned_data.pkl'):
        """
        Args:
            path (string): Path to the .pkl file that contains all the 
            data 
        """
        self.path = path
        self.dataset = self.extract_info() # list of matrices
        self.validation = False
        self.lookup = MoleType

    def __len__(self):
        '''
        Returns:
            total: sums over the length of the individual data sets
        '''
        return len(self.dataset)
        
    def __getitem__(self,idx):
        img_path, label = self.dataset[idx] 
        img = Image.open(img_path)
        label = self.lookup[label]
        return img, label 
        
    def extract_info(self): 
        return pickle.load(open(self.path,'rb'))
         
 
def main(): 
    print('--- Running Test for ImageDataset class ---')
    dataset = ImageDataset()
    print('Length of Dataset', len(dataset))  
    index = 100 
    print('Returning Item at index: ', index)
    print(dataset[index])   
 
if __name__ == '__main__': 
   main()