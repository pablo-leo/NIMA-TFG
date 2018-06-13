# -------------- Imports --------------------------- 

# packages needed
import numpy as np
import pandas as pd

# keras imports
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

# allow memmory dynamic memmory allocation
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# model selection
from sklearn.model_selection import train_test_split

# to get image names
import os, cv2, random
import ntpath

from BatchGenerator import BatchCreator, BatchSequence
from create_model import create_VGG
from data_utils import load_ava, clean_data

# random seed to make the kernel reproducible
random_seed = 123456789
np.random.seed(random_seed)


# -------------- Data load / clean / split --------------------------- 


# directory where we are going to work on
workdir = '/home/frubio/AVA/'

# create result dir
result_dir = './results' # define here the directory where your results will be saved
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# load the data
x_train, y_train = load_ava(workdir)

# clean the data
x_train, y_train = clean_data(workdir, x_train, y_train)

# define directory of training images
train_dir_images = os.path.join(workdir, 'AVA_images')

# split train and validationy_train
x_train, x_validation, y_train, y_validation = train_test_split(x_train, 
                                                                y_train, 
                                                                test_size = 0.2,
                                                                random_state = random_seed)

# -------------- Loss function --------------------------- 

def emd_loss(y_true, y_pred):
    '''
    Earth Mover's Distance loss
    '''
    cdf_p    = K.cumsum(y_true, axis = -1)
    cdf_phat = K.cumsum(y_pred, axis = -1)
    loss     = K.mean(K.sqrt(K.mean(K.square(K.abs(cdf_p - cdf_phat)), axis = -1)))
    
    return loss


# -------------- Model load and training --------------------------- 

vgg_16 = create_VGG()

# learning rate
learning_rate = 1e-3

# optimizer and model compile
optimizer = Adam(lr = learning_rate)
vgg_16.compile(optimizer, emd_loss)

# training
checkpoint = ModelCheckpoint(os.path.join(result_dir, 'best_model_{val_loss:.2f}.h5'), 
                             monitor = 'val_loss',
                             verbose = 1,
                             save_weights_only = True,
                             save_best_only = True,
                             mode = 'min')

# batch size and number of epochs
batch_size = 150
epochs = 10

# create the Batches and obtain the generatoros
training_data = BatchCreator(x_train, y_train, batch_size, train_dir_images)
validation_data = BatchSequence(x_validation, y_validation, batch_size, train_dir_images)

# fit the model
vgg_16.fit_generator(training_data,
                     steps_per_epoch = len(training_data),
                     epochs = epochs,
                     verbose = 1, 
                     callbacks = [checkpoint],
                     validation_data = validation_data,
                     validation_steps = len(validation_data),
                     workers = 4,
                     use_multiprocessing = True)
