# -------------- Imports --------------------------- 
from argparse import ArgumentParser
from os import listdir
from os.path import join, isfile, isdir
from utils.create_model import create_VGG, create_MobileNet, create_Inception
from utils.score_utils import mean_score, std_score
import sys, cv2
import numpy as np
import pandas as pd

# ------ Function that generate the scores ---------
def generate_scores(parser):
    
    # get the model name
    model_name = parser.net
    
    # check folder 
    if parser.file is None and parser.folder is None:
        raise Exception('Indicate a file or a folder path')
    
    # list the files to evaluate
    if parser.folder is not None and isdir(parser.folder):
        files = [join(parser.folder, file) for file in listdir(parser.folder)]
    else:
        files = []
    
    # check if the file exists
    if parser.file is not None and len(files) == 0 and isfile(parser.file):
        files = [parser.file]
    
    # if there is not folder or file raise exception
    if len(files) == 0:
        raise Exception('File or folder doesn\'t exists')
    
    # print the files to evaluate
    if parser.vb == 1:
        print('---------- Files to evaluate ----------')
        for i in range(len(files)):
            print('{}: {}'.format(i, files[i]))
        print('---------------------------------------')

    # create vgg, mobilenet or inception
    if model_name == 'vgg_16':
        model = create_VGG()
    elif model_name == 'mobilenet':
        model = create_MobileNet()
    elif model_name == 'inception':
        model = create_Inception()
    
    # load the weights of the network
    model.load_weights('./weights/{}.h5'.format(model_name))
    
    images = []
    # load the images into memmory
    for file in files:
        image = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))/255
        images.append(image)
        
    # convert list into numpy array
    images = np.array(images)
    
    y_pred = model.predict(images)
    mean_scores = [mean_score(pred) for pred in y_pred]
    std_scores = [std_score(pred) for pred in y_pred]
    
    print('-------------- Evaluation --------------')
    for i in range(len(files)):
        score = '{} ({}{})'.format(round(mean_scores[i], 3), chr(177), round(std_scores[i], 3))
        print('{}: {} \n\t{}'.format(i, files[i], score))
    print('----------------------------------------')
    
    # save the predictions
    if parser.save:
        data = np.stack([files, mean_scores, std_scores], axis = 1)
        df = pd.DataFrame(data = data, columns = ['file', 'mean', 'std'])
        df.to_csv('evaluations.csv', index = False)

if __name__ == '__main__':
    p = ArgumentParser('Neural Image Assesment evaluate')

    # variables
    p.add_argument('-net', type = str, default = 'inception',  choices=['vgg_16', 'inception', 'mobilenet'],
                   help = 'network used for evaluating (default: inception)')
    p.add_argument('-file', type = str, default = None,
                   help = 'path of the image that will be evaluated')
    p.add_argument('-folder', type = str, default = None,
                   help = 'path of the folder containing the files')
    p.add_argument('-vb', type = int, default = 1, choices=[0, 1], 
                   help = 'print information (default: 1)')
    p.add_argument('-save', type = int, default = 0,  choices=[0, 1],
                   help = 'save the evaluations to a csv file (default: 0)')

    generate_scores(p.parse_args())
    sys.exit(0)