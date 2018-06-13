import os
import pandas as pd
import numpy as np

def load_ava(workdir):
    
    # set the correct directory
    ava_dir = os.path.join(workdir, 'AVA.txt')
    ava_data_pd = pd.read_csv(ava_dir, sep = ' ', header = None, names = ['Index', 'Image ID', 'Rate 1', 'Rate 2',
                                                                          'Rate 3', 'Rate 4', 'Rate 5', 'Rate 6', 
                                                                          'Rate 7', 'Rate 8', 'Rate 9','Rate 10', 
                                                                          'Sem 1', 'Sem 2', 'Challenge ID'])

    # separate the labels of the image IDs
    x_train = ava_data_pd.values[:,1]
    y_train = ava_data_pd.values[:,2:12]
    
    return x_train, y_train

def clean_data(workdir, x, y):
    
    # define directory of training images
    train_dir_images = os.path.join(workdir)

    # list containing the ids of the images that are not in the directory
    lost_images = []
    #lost_images_aux = []

    current_images = [filename[:-4] for filename in os.listdir(train_dir_images)]
    lost_mask = np.isin(x, current_images)

    # prints number of lost images and original shape
    print("Total images lost: ", len(lost_mask) - np.sum(lost_mask))
    print("Original dataset shape: ", x.shape)

    x = x[lost_mask]
    y = y[lost_mask]

    # resulting shape
    print("Resulting dataset shape: ", x.shape)
    
    return x, y