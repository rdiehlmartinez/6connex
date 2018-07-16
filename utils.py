__author__ = 'Richard Diehl Martinez'

import cv2
from torchvision import transforms
from PIL import Image
from enum import Enum

'''
utils.y
--------
Utils file that provides useful functions to be used in the
demonstration of the classification model.
'''

class MoleType(Enum):
    benign = 0
    malignant = 1
    indeterminate = 2

def return_prediction_label(prediction_tensor):
     _, index = prediction_tensor.max(1)
     index = index.item()
     return MoleType(index).name

def process_img(img):
    img = Image.fromarray(img)
    trans = transforms.ToTensor()
    resize = transforms.Resize((144,192))
    return trans(resize(img)).unsqueeze(0)

def capture_screenshot():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            print('Exited before capturing image')
            exit()
        elif key == 32: # spacebar
            cv2.imwrite('demo_img.jpg',frame)
            print('Captured Skin Image')
            return frame

    cv2.destroyWindow("preview")

if __name__ == '__main__':
    capture_screenshot()
