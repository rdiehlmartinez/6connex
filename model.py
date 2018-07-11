__author__ = 'Richard Diehl Martinez' 

'''
Establishes the basic training process 
which which to train the model for classification of 
images into benign/malignant/indeterminant.
'''

from dataset import ImageDataset
from classifier import ClassificationModel 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import torch.nn.functional as F
import numpy as np
import argparse


class Model(): 
    def __init__(self, lr, batchsize, verbose, dtype = torch.float32): 
        """
        Model to initialize the training of the convolutional neural network. 
        @ Arguments: 
            dtype: type of the values used in the model 
            device: device to store the model on, default(cpu)
        """
        
        # Config Training
        self.dtype = dtype
        self.num_epochs = 30
        self.lr = lr
        self.batch_size = 128
        self.verbose = verbose # Boolean 
        
        # Split Between Training and Validation Datasets
        self.split_prop = 0.9

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
        self.split_index = int(N*self.split_prop)
        print("Successfully Loaded Dataset, With {} Examples Total".format(N))
        return dataset

    def get_batcher(self, dataset, eval = False):
        if eval: 
            sampler = SubsetRandomSampler(np.arange(self.split_index, len(dataset)))
        else: 
            sampler = SubsetRandomSampler(np.arange(self.split_index))
        batcher = DataLoader(dataset,sampler=sampler,batch_size=self.batch_size)
        return batcher

    def normalize_batch(self, inputs):
        inputs -= inputs.mean(dim=1)
        inputs /= inputs.std(dim=1)
        return inputs
        
    def train(self, eval = True):
        """
        Trains the model using the config values passed into the model. If the 
        validation flag is set to true, prints out the validation loss and 
        training loss. The validation loss is only printed out after 
        every epoch. 
        
        The model uses: 
            - Cross Entropy Loss with a final softmax layer 
            - Adam Optimizer is used with default beta parameters and 
              user specified learning rate  
        """
        
        self.model = self.model.to(device = self.device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr)
        dataset = self.get_dataset()
        training_batcher = self.get_batcher(dataset)
        global_iteration = 0
        prev_loss = None
        
        if self.verbose:
            print_every = 2
        else:
            print_every = 10
        
        for epoch in range(self.num_epochs):
            print("Entering epoch number: {}".format(epoch))
            avg_loss = torch.tensor([0.0]).to(device=self.device)
            self.model = self.model.train() 
            
            for iteration, batch in enumerate(training_batcher):
                global_iteration += 1 
                inputs, labels = batch 

                #with torch.no_grad():
                #    normalized_inputs = normalize_batch(inputs)

                optimizer.zero_grad()
                outputs = self.model(normalized_inputs)
                
                loss = (torch.nn.CrossEntropyLoss().to(device = self.device))(outputs, labels)
                self.accuracy(outputs,loss)
                exit()
                loss.backward()
                optimizer.step()

                if(global_iteration % print_every == 0):
                    loss = loss.item()
                    print('Iteration %d, Loss = %.4f' % (global_iteration,loss))
                    
            if eval: 
                # Evaluating the validation performance of the model
                self.model = self.model.eval()
                validation_batcher = self.get_batcher(dataset, eval = True)
                for iteration, batch in enumerate(training_batcher):
                    inputs, labels = batch
                    predictions = model(input)
                
        print('Finished Training')
        
    def accuracy(self,predictions, labels): 
        '''
        Returns the accuracy over a batch of predictions and labels. 
        @args
        predictions (3 dimensional matrix): output unnormalized probs of the model 
        labels (1 dimensional matrix): correct label for the dataset 
        @returns (scalar): averaged accuracy over the batch
        '''
        print(predictions.shape)
    
        
    def predict(self, input, eval=False):
        """
        Performs inference. 
        @args
        input_image (tensor): A tensor of size (N, C, H, W), representing a batch of images
                              to perform inference on.
        @returns
        A tensor of size (N, 3), representing the predicted label
        """
        model = self.model.to(device = self.device)
        input = input.to(device=self.device,dtype=self.dtype)
        predictions = model(input)
        return predictions


def main():
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('lr', type=float, nargs=None, help='initial learning rate for model', metavar='initial learning rate')
    parser.add_argument('batch_size', type=int, nargs=None, help='batch size for model', metavar='batch size')
    parser.add_argument("-v", "--verbose", help='show more info', action="store_true")
    args = parser.parse_args()
    
    model = Model(args.lr, args.batch_size, args.verbose)
    model.train()

if __name__ == '__main__':
    main()
