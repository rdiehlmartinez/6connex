__author__ = 'Richard Diehl Martinez'
'''
File to analyze the dataset, and to print out relevant statistics.
'''

import os 
import json
import pickle

PATH = 'ISIC-images' 
OUTFILE = 'cleaned_data.pkl'

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
    
def get_stats(pic_filenames): 
    
    '''
    Prints relevant statistics of the dataset. Like the proportion of 
    entries that are of a certain label.
    '''
    
    assert(len(pic_filenames) > 0)
    label_counts = {'benign':0,'malignant':0,'indeterminate':0}
    for pic_file in pic_filenames:
        json_file = pic_file.replace('.jpg', '.json')
        try:   
            curr_data = json.loads(open(json_file).read())
            curr_label = curr_data['meta']['clinical']['benign_malignant']
            if curr_label in valid_labels: 
                label_counts[curr_label] = label_counts[curr_label] + 1 
        except:
            continue 
    print(label_counts)
        

def main():
    pic_filenames, json_filenames = return_filenames()
    print('Length raw data: ', len(pic_filenames))
    get_stats(pic_filenames)

if __name__ == '__main__':
    main()