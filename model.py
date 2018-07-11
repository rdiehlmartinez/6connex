__author__ = 'Richard Diehl Martinez' 

'''
Establishes the basic CNN model for classification of 
images into benign/malignant/indeterminant.
'''

from dataset import ImageDataset
from classifier import ClassificationModel 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.nn.functional as F
import numpy as np
import argparse


class Model(): 
    def __init__(self, lr_in, batchsize_in, dtype = torch.float32, device = torch.device('cpu')): 
        
        # Config Training
        self.dtype = dtype
        self.num_epochs = 30
        self.lr = lr_in                        
        self.device = device                        
        self.batch_size = batchsize_in

        # Config Model
        self.num_outputs = 3
        
        # Initialize CNN Model
        self.model = ClassificationModel(num_outputs=self.num_outputs)

        if (torch.cuda.is_available()):
            num_GPUs = torch.cuda.device_count()
            print("Targeting {} GPU(s)".format(num_GPUs))
            self.device = torch.device('cuda:0')

            if (num_GPUs > 1):
                self.model = nn.DataParallel(self.model)
        else:
            print("CUDA Not Available")
            print("Targeting CPU")
            self.device = torch.device('cpu')                  
            
    def trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        
    def get_dataset(self, path ='cleaned_data.pkl'):
        dataset = ImageDataset(path)
        N = len(dataset)
        print("Successfully Loaded Dataset, With {} Examples Total".format(N))
        return dataset, N

    def get_batcher(self, dataset):
        sampler = RandomSampler(dataset)
        batcher = DataLoader(dataset,sampler=sampler,batch_size=self.batch_size)
        return batcher

    def normalize_batch(self, inputs):
        inputs -= inputs.mean(dim=1)
        inputs /= inputs.std(dim=1)
        return inputs
        
    def train(self):
        self.model = self.model.to(device = self.device)              #wat
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr)
        dataset, N = self.get_dataset()
        training_batcher = self.get_batcher(dataset)
        global_iteration = 0
        prev_loss = None
        
        for epoch in range(self.num_epochs):
            print("Entering epoch number: {}".format(epoch))
            avg_loss = torch.tensor([0.0]).to(device=self.device)
            predictions, output_labels = None, None
            for iteration, batch in enumerate(training_batcher):
                
                ''' 
                Uncomment to check size of batches 
                print(len(batch))
                print(batch[0].shape)
                print(batch[1].shape)
                exit() 
                '''
                
                
                global_iteration += 1 
                
                inputs, labels = batch 

                with torch.no_grad():
                    normalized_inputs = normalize_batch(inputs)

                optimizer.zero_grad()
                outputs = self.model(normalized_inputs)
                
                loss = (torch.nn.CrossEntropyLoss().to(device = self.device))(outputs, labels)
                loss.backward()
                optimizer.step()

                if(global_iteration % 1 == 0):
                    loss = loss.item()
                    print('Iteration %d, Loss = %.4f' % (global_iteration,loss))
                
        print('Finished Training')


def main():
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('lr', type=float, nargs=None, help='initial learning rate for model', metavar='initial learning rate')
    parser.add_argument('batch_size', type=int, nargs=None, help='batch size for model', metavar='batch size')
    parser.add_argument("-v", "--verbose", help='show more info', action="store_true")

    args = parser.parse_args()
    verbose = args.verbose
    model = Model(args.lr, args.batch_size)
    model.train()

if __name__ == '__main__':
    main()
