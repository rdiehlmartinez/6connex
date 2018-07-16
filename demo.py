__author__ = 'Richard Diehl Martinez'

from utils import capture_screenshot, process_img, return_prediction_label
import torch
from PIL import Image
import torch
import torch.nn as nn
from classifier import ClassificationModel

'''
demo.py
--------
Demo for the classification algorithm of skin conditions into malignant or
benign. This demo uses the built-in computer camera to take a picture of
the skin area in question. This picture is then fed into the machine
learning model which then predictes whether the patch of skin is
dangerous or not.
'''

ascii_art = """
  ________
 /  _____/ ____  ____   ____   ____   ____ ___  ___
/   __  \_/ ___\/  _ \ /    \ /    \_/ __ \\  \/  /
\  |__\  \  \__(  <_> )   |  \   |  \  ___/ >    <
 \_____  /\___  >____/|___|  /___|  /\___  >__/\_ /
"""

model_path = 'saved_models/acc_0.75.pt'

def predict(input, model):
        """
        Performs inference.
        @args
        input_image (tensor): A tensor of size (N, C, H, W), representing a batch of images
                              to perform inference on.
        @returns
        The predicted label
        """
        model = model.eval()
        predictions = model(input) # returns a tensor
        prediction_label = return_prediction_label(predictions)
        return prediction_label

def main():
    print(ascii_art)
    print('Welcome to the demo for the skin classification algorithm.')
    print(10*'--')
    print('To begin please hit the spacebar to capture an area of skin.')
    input_image = capture_screenshot()
    input = process_img(input_image)
    print(10*'--')
    print('Succesfully read in input image.')
    print('')
    print('Running classification algorithm...')
    model = ClassificationModel()
    model.load_state_dict(torch.load(model_path))
    prediction = predict(input,model)
    print('This sample of skin looks: ', prediction)

if __name__ == '__main__':
    main()
