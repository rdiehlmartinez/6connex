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
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import argparse
import shutil

class Model(): 
    def __init__(self, lr, batch_size, split_prop = 0.9, verbose = False, dtype = torch.float32): 
        """
        Model to initialize the training of the convolutional neural network. 
        @ Arguments: 
            lr: The learning rate for the model 
            batch_size: The initial batch_size to be used in the model 
            split_prop: The proportion of data used in the training versus validation 
                        dataset; when split_prop = 1 all of the data is trained on 
            verbose: boolean to specify the frequency with which to print out loss and accuracy values 
            dtype: type of the values used in the model 
        """
        
        # Config Training
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose # Boolean 
        self.dtype = dtype 
        self.num_epochs = 25
        
        # Split Between Training and Validation Datasets
        self.split_prop = split_prop
        
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
        '''
        Helper function to print out the number of parameters in our model.
        '''
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        
    def get_dataset(self, path ='cleaned_data.pkl'):
        '''
        Returns a dataset object, which we define in the dataset.py 
        file.  
        '''
        dataset = ImageDataset(path)
        N = len(dataset)
        self.split_index = int(N*self.split_prop)
        print("Successfully Loaded Dataset, With {} Examples Total".format(N))
        return dataset

    def get_batcher(self, dataset, eval = False):
        """
        Returns a 'batcher' to the training subroutine which 
        trains the machine learning model. Notice that this batcher
        is simply a PyTorch DataLoader Object. We pass into this 
        dataloader both the dataset (which is a class we define) and 
        a sampler. The sampler is another object which informs the 
        dataloader how it should sample batches from the dataset. 
        
        By specifying that the sampler should only sample form certain
        indices of the dataset, we can effectively create a training-validation
        split of our data.
        
        @args: 
            dataset (ImageDataset) 
        @returns:
            batcher (DataLoader)
            
        
        """
        if eval: 
            sampler = SubsetRandomSampler(np.arange(self.split_index, len(dataset)))
        else: 
            sampler = SubsetRandomSampler(np.arange(self.split_index))
        batcher = DataLoader(dataset,sampler=sampler,batch_size=self.batch_size)
        return batcher

    def normalize_batch(self, inputs):
        '''
        Normalizes the batch inputs in the training process. 
        @args: 
            Unnormalized Images
        @returns:
            Normalized images 
        '''
        inputs -= inputs.mean(dim=0,keepdim=True)
        inputs /= inputs.std(dim=0,keepdim=True)
        return inputs
        
    def train(self):
        """
        Trains the model using the config values passed into the model. If a
        training/validation loss is specified, prints out the validation loss and 
        training loss. The validation accuracy is only printed out after 
        every epoch. 
        
        The model uses: 
            - Cross Entropy Loss with a final softmax layer 
            - Adam Optimizer is used with default beta parameters and 
              user specified learning rate  
        """
        
        self.model = self.model.to(device = self.device)
        print('Training model with {} parameters'.format(str(self.trainable_parameters())))
        eval = (self.split_prop < 1)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr)
        dataset = self.get_dataset()
        
        # Splitting Data between Training and Validation
        training_batcher = self.get_batcher(dataset)
        if eval: 
            validation_batcher = self.get_batcher(dataset, eval = True)
            
        global_iteration = 0
        prev_loss = None
        
        if self.verbose:
            print_every = 1
        else:
            print_every = 10
        
        best_train_acc = 0 
        best_val_acc = 0
        
        for epoch in range(self.num_epochs):
            print("Entering epoch number: {}".format(epoch))
            avg_loss = torch.tensor([0.0]).to(device=self.device)
            self.model = self.model.train() 
            
            for iteration, batch in enumerate(training_batcher):
                global_iteration += 1 
                inputs, labels = batch 

                with torch.no_grad():
                    normalized_inputs = self.normalize_batch(inputs).to(device = self.device)

                optimizer.zero_grad()
                outputs = self.model(normalized_inputs)
                
                loss = (torch.nn.CrossEntropyLoss().to(device = self.device))(outputs, labels)
                loss.backward()
                optimizer.step()

                if(global_iteration % print_every == 0):
                    loss = loss.item()
                    accuracy = self.accuracy(outputs,labels)
                    if(accuracy > best_train_acc): 
                        best_train_acc = accuracy
                    print('Global iteration %d, Loss = %.4f, Training Accuracy = %.4f' % (global_iteration,loss, accuracy))
                    
            if eval: 
                # Evaluating the validation performance of the model
                self.model = self.model.eval()
                curr_val_acc = 0 
                for iteration, batch in enumerate(validation_batcher):
                    inputs, labels = batch
                    predictions = self.model(inputs).to(device = self.device)
                    curr_val_acc += self.accuracy(predictions,labels)
                
                curr_val_acc /= (iteration+1) # Number of batch iterations 
                print('Epoch %d, Validation Accuracy = %.4f' % (epoch,curr_val_acc))
                if(curr_val_acc > best_val_acc):
                    print('New Best Validation Accuracy!') 
                    best_val_acc = curr_val_acc
                    self.save_model_params(self.model,curr_val_acc)
                        
        print('Model Finished Training')
        
        
    def save_model_params(self, model, accuracy): 
        '''
        Saves out the model parameters only. Call on this method if the model 
        should only be used for future predictions (such as for demos). 
        @args: 
            model (ClassificationModel): the model whose parameters should be saved 
        '''
        print('Saving Model parameters')
        file_name = "saved_models/acc_{}.pt".format(str(accuracy))
        torch.save(model.state_dict(), file_name)
        
    def save_checkpoint(self, state, is_best=False): 
        '''
        Saves a checkpoint of the model which can be used to continue training of the 
        model. The model should only be saved in this format if we hope to continue 
        training the model. This method is more comprehensive than the save_model_param
        method.
        @args:
            state (dictionary): A state dictionary should be passed in which 
            contains the following values
                {
                    epoch: current Epoch 
                    state_dict: current state of the model parameters
                    optimizer: optimizer used for training 
                    best_acc: current best validation accuracy 
                }
                
            is_best (boolean): boolean to determine if the current model is the best so far
        
        '''
        file_name = "checkpoints/acc_{}.pth.tar".format(str(state['best_acc']))
        torch.save(state,file_name)
        if(is_best): 
            print('New best validation accuracy, saving file')
            shutil.copyfile(file_name, 'model_best.pth.tar')
        
        
    def accuracy(self,predictions, labels): 
        '''
        Returns the accuracy over a batch of predictions and labels. 
        @args
        predictions (3 dimensional matrix): output unnormalized probs of the model 
        labels (1 dimensional matrix): correct label for the dataset 
        @returns (scalar): averaged accuracy over the batch
        '''
        _, predicted_indices = predictions.max(1)
        N = float(predicted_indices.shape[0])
        accuracy = torch.sum(predicted_indices == labels).item()/N
        return accuracy
        
        
    def predict(self, input, eval=False):
        """
        Performs inference. 
        @args
        input_image (tensor): A tensor of size (N, C, H, W), representing a batch of images
                              to perform inference on.
        @returns
        A tensor of size (N, 3), representing the predicted label
        """
        if eval:
            self.model = self.model.eval()
            
        model = self.model.to(device = self.device)
        input = input.to(device=self.device,dtype=self.dtype)
        predictions = model(input)
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Run the model.')

    parser.add_argument('-lr', action = 'store', dest = 'lr', type=float, nargs=None, help='initial learning rate', metavar='initial learning rate')
    parser.add_argument('-batch_size', action = 'store', dest = 'bs', type=int, nargs=None, help='batch size', metavar='batch size')
    parser.add_argument("-v", "--verbose", help='set flag to print out model results every iteration; otherwise printed out every 5 iterations', action="store_true")
    parser.add_argument("-split_prop", "--split_prop", help='specify the proportion of data to split into training/test set, defaulted to 0.9', type = float, action="store", default=0.9)
    args = parser.parse_args()
    assert(args.lr is not None and args.bs is not None)
        
    model = Model(args.lr, args.bs, split_prop = args.split_prop, verbose = args.verbose)
    model.train()

if __name__ == '__main__':
    main()
