__author__ = 'Richard Diehl Martinez'
'''
File to analyze the dataset, and to print out relevant statistics.
'''

import os 
import json
import pickle
from PIL import Image

PATH = 'ISIC-images' 
OUTFILE = 'cleaned_data.pkl'
OUTFILE_LARGE = 'cleaned_data_large.pkl'

valid_labels = ['benign','malignant','indeterminate']

def return_filenames(path = PATH):
    '''
    Returns the filenames of all of the images and json 
    objects
        @Args:
            path: name of folder where data is stored 
        @Returns 
            pics: list of filenames of images 
            info: list of filenames of json objects 
    '''
    
    pics = []
    jsons = []
    for root, dirs, files in os.walk(path):  
        for filename in files:
            if(".json" in filename):
                jsons.append(os.path.join(root,filename))
            elif(".jpg" in filename): 
                pics.append(os.path.join(root,filename))
    return((pics,jsons))
    
def clean_data(pic_filenames): 
    '''
    Returns a list of tuples that matches all of the 
    images in the dataset with the corresponding label
    (benign, malicious). This provides a compressed version 
    of the cleaned dataset. 
        @Args:
            pics: list of filenames of images 
        @Returns 
            data: returns the matched and cleaned data (pic_file, malignant/benign)
    '''
    
    assert(len(pic_filenames) > 0)
    data = [] 
    for pic_file in pic_filenames:
        json_file = pic_file.replace('.jpg', '.json')
        try:   
            curr_data = json.loads(open(json_file).read())
            curr_label = curr_data['meta']['clinical']['benign_malignant']
            if curr_label in valid_labels:
                data.append((pic_file,curr_label))
        except:
            continue 
    return data
        
def clean_data_store_images(pic_filenames):
    '''
    Returns a list of tuples that matches all of the 
    images in the dataset with the corresponding label
    (benign, malicious). This method actually stores the 
    images in the list, not just their path names. 
        @Args:
            pics: list of filenames of images 
        @Returns 
            data: returns the matched and cleaned data (pic, malignant/benign)
    '''
    
    assert(len(pic_filenames) > 0)
    data = [] 
    for pic_file in pic_filenames:
        json_file = pic_file.replace('.jpg', '.json')
        try:   
            curr_data = json.loads(open(json_file).read())
            curr_label = curr_data['meta']['clinical']['benign_malignant']
            pic = Image.open(pic_file)
            if curr_label in valid_labels:
                data.append((pic,curr_label))
        except:
            continue 
    return data

def main():
    pic_filenames, json_filenames = return_filenames()
    print('Length raw data: ', len(pic_filenames))
    #data = clean_data(pic_filenames)
    data = clean_data_store_images(pic_filenames)
    print('Length of cleaned data: ', len(data))
    with open(OUTFILE_LARGE, 'wb') as fp:
        pickle.dump(data, fp)


if __name__ == '__main__':
    main()