# -------------- Imports --------------------------- 
from argparse import ArgumentParser
from os.path import exists, join
from os import mkdir
from sklearn.model_selection import train_test_split
from utils.BatchGenerator import BatchCreator, BatchSequence
from utils.create_model import create_VGG, create_MobileNet, create_Inception
from utils.data_utils import load_ava, clean_data
from utils.score_utils import emd_loss
from utils.callbacks import PlotLogs
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import sys

# ------ Function that trains the models ---------
def train(parser):
    
    # get the model name
    model_name = parser.net
    
    # get the directory of the annotations and the images
    train_dir_annotations = parser.andir
    train_dir_images = parser.imdir
    
    # create result dir to store the models
    result_dir = './' + parser.res
    if not exists(result_dir):
        mkdir(result_dir)
    
    # load the AVA.txt anotations
    x_train, y_train = load_ava(train_dir_annotations)
    
    # delete lost entries from the dataset
    x_train, y_train = clean_data(train_dir_images, x_train, y_train)
    
    # split the dataset into validation and training
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, 
                                                                    y_train, 
                                                                    test_size = parser.tsize,
                                                                    random_state = parser.rseed)
    
    

    # create vgg, mobilenet or inception
    if model_name == 'vgg_16':
        model = create_VGG(verbose = 2)
    elif model_name == 'mobilenet':
        model = create_MobileNet(verbose = 2)
    elif model_name == 'inception':
        model = create_Inception(verbose = 2)
        
    # learning rate
    learning_rate = parser.lr

    # optimizer and model compile
    optimizer = Adam(lr = learning_rate, decay = parser.dc)
    model.compile(optimizer, emd_loss)

    # load or not the model
    load_checkpoint = parser.checkpoint
    model_filename = './results/best_model_{}.h5'.format(model_name)
    
    # training chekpoints
    checkpoint = ModelCheckpoint(join(result_dir, 'best_model_{}.h5'.format(model_name)), 
                                 monitor = 'val_loss',
                                 verbose = 1,
                                 save_weights_only = True,
                                 save_best_only = True,
                                 mode = 'min')

    # plot the logs
    plotlogs = PlotLogs(model_filename)

    # batch size and number of epochs
    batch_size = parser.bsize
    epochs = parser.epochs

    # create the Batches and obtain the generatoros
    training_data = BatchCreator(x_train, y_train, batch_size, train_dir_images)
    validation_data = BatchSequence(x_validation, y_validation, batch_size, train_dir_images)
    
    # load past checkpoint
    if load_checkpoint and exists(model_filename):
        model.load_weights(model_filename)
        model.evaluate_generator(validation_data,
                                 verbose = 1,
                                 workers = 4,
                                 use_multiprocessing = True)
        print('weights loaded')

    # fit the model
    model.fit_generator(training_data,
                         steps_per_epoch = len(training_data)//2,
                         epochs = epochs,
                         verbose = 1, 
                         callbacks = [checkpoint, plotlogs],
                         validation_data = validation_data,
                         validation_steps = len(validation_data),
                         workers = 4,
                         use_multiprocessing = True)

if __name__ == '__main__':
    p = ArgumentParser('Neural Image Assesment train')

    # variables
    p.add_argument('-net', type = str, default = 'inception',  choices=['vgg_16', 'inception', 'mobilenet'],
                   help = 'network used for training (default: inception)')
    p.add_argument('-res', type = str, default = 'results',
                   help = 'name of the folder to store the results (default: results)')
    p.add_argument('-andir', type = str, default = None,
                   help = 'path of the directory containing the AVA.txt file')
    p.add_argument('-imdir', type = str, default = None,
                   help = 'path of the directory containing all the images')
    p.add_argument('-rseed', type = int, default = 123456789,
                   help = 'seed for generating the val/train split (default: 123456789)')
    p.add_argument('-tsize', type = float, default = 0.2,
                   help = 'validation set size (default: 0.2)')    
    p.add_argument('-lr', type = float, default = 3e-6,
                   help = 'learning rate for the ADAM optimizer (default: 3e-6)')
    p.add_argument('-dc', type = float, default = 1e-8,
                   help = 'learning rate decay for the ADAM optimizer (default: 1e-8)')
    p.add_argument('-bsize', type = int, default = 100,
                   help = 'batch size (default: 100)')
    p.add_argument('-epochs', type = int, default = 20,
                   help = 'number of epochs (default: 20)')
    p.add_argument('-checkpoint', type = int, default = 0, choices=[0, 1], 
                   help = 'load previous checkpoint (default: 0)')
    p.add_argument('-vb', type = int, default = 1, choices=[0, 1], 
                   help = 'print information (default: 1)')

    train(p.parse_args())
    sys.exit(0)