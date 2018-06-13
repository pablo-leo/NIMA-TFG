import numpy as np
import os, cv2
from keras.utils import Sequence

class BatchCreator(object):

    def __init__(self, x, y, batch_size, images_dir):
        '''
        x: numpy array containing the Image IDs
        y: numpy array containing the rating distribution
        batch_size: desired batch size
        images_dir: directory of the images
        '''
        
        # params
        self.imgs_ids        = x                # image IDs
        self.imgs_rates      = y                # image rate distribution
        self.batch_size      = batch_size       # number of patches per batch
        self.images_dir      = images_dir       # directory containing the images
        
        # batch information
        self.n_samples = len(self.imgs_ids)
        self.n_batches = self.n_samples // self.batch_size
    
        # information print
        print('BatchCreator: {n_samples} patch samples.'.format(n_samples=self.n_samples))
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        # length as number of batches
        return self.n_batches

    def next(self):
        # build a mini-batch

        batch_x = np.zeros((self.batch_size, 224, 224, 3), dtype = np.float32)
        batch_y = np.zeros((self.batch_size, 10), dtype = np.float32) # one-hot encoding
                
        # load and return "batch_size" images
        for i in range(self.batch_size):

            # select case (generate random index)
            img_index = np.random.randint(len(self.imgs_ids))

            # obtain the path
            case = os.path.join(self.images_dir, str(self.imgs_ids[img_index]) + '.jpg')

            # load image, change color structure and resize it
            img = cv2.resize(cv2.cvtColor(cv2.imread(case, 1), cv2.COLOR_BGR2RGB), (256, 256))/255
            
            # get croping starting points
            x, y = np.random.randint(0,32,2)
            
            # store cropped image
            batch_x[i] = img[x:x+224, y:224+y]
            
            # store "normalized" score distribution
            batch_y[i] = self.imgs_rates[img_index]/np.sum(self.imgs_rates[img_index])
                        
        return batch_x, batch_y


class BatchSequence(Sequence):

    def __init__(self, x, y, batch_size, images_dir):
        '''
        x: numpy array containing the Image IDs
        y: numpy array containing the rating distribution
        batch_size: desired batch size
        images_dir: directory of the images
        '''
        
        # params
        self.imgs_ids   = x          # image IDs
        self.imgs_rates = y          # image rate distribution
        self.batch_size = batch_size # number of patches per batch
        self.images_dir = images_dir # directory containing the images
        
        # length
        self.n_samples = len(self.imgs_ids)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))  # last mini-batch might be shorter
        
        # print some info
        print('PatchSequence: {n_samples} patch samples.'.format(n_samples=self.n_samples))

    def __len__(self):
        # provide length in number of batches
        return self.n_batches
    
    def __getitem__(self, idx):

        # create indexes for samples
        idx1 = idx * self.batch_size
        idx2 = np.min([idx1 + self.batch_size, self.n_samples]) # last batch
        idxs = np.arange(idx1, idx2)
        
        batch_x = np.zeros((len(idxs), 224, 224, 3), dtype = np.float32)
        batch_y = np.zeros((len(idxs), 10), dtype = np.float32) # one-hot encoding

        # load and return "batch_size" images
        for i in range(len(idxs)):

            # obtain the path
            case = os.path.join(self.images_dir, str(self.imgs_ids[idxs[i]]) + '.jpg')

            # load image, change color structure, resize it and store it
            batch_x[i] = cv2.resize(cv2.cvtColor(cv2.imread(case, 1), cv2.COLOR_BGR2RGB), (224, 224))/255

            # store score distribution
            batch_y[i] = self.imgs_rates[idxs[i]]/np.sum(self.imgs_rates[idxs[i]])

        return batch_x, batch_y