from keras.applications import vgg16, mobilenet
from keras.layers import Dropout, Dense
from keras.models import Model

def create_VGG(dropout_rate = 0.75):
    
    # set the image input size
    image_size = 224
    
    # load the pre-trained model
    vgg_16 = vgg16.VGG16(include_top = True, weights='imagenet', input_shape = (image_size, image_size, 3))

    # remove last layer
    vgg_16.layers.pop()

    # freeze all the layers
    for layer in vgg_16.layers:
        layer.trainable = False

    # get the output of the model
    x = vgg_16.layers[-1].output

    # add last FC layer
    x = Dropout(dropout_rate)(x)
    x = Dense(10, kernel_initializer = 'he_uniform', activation = 'softmax')(x)

    # create the new model
    vgg_16 = Model(vgg_16.input, x)

    # show summary
    vgg_16.summary()
    
    return vgg_16

def create_MobileNet(dropout_rate = 0.75):
    
    # set the image input size
    image_size = 224
    
    # load the pre-trained model
    mobile_net = mobilenet.MobileNet(include_top = True, weights='imagenet', input_shape = (image_size, image_size, 3))

    # remove last layer
    mobile_net.layers.pop()
    mobile_net.layers.pop()
    mobile_net.layers.pop()

    # freeze all the layers
    for layer in mobile_net.layers:
        layer.trainable = False

    # get the output of the model
    x = mobile_net.layers[-1].output

    # add last FC layer
    x = Dropout(dropout_rate)(x)
    x = Dense(10, kernel_initializer = 'he_uniform', activation = 'softmax')(x)

    # create the new model
    mobile_net = Model(mobile_net.input, x)

    # show summary
    mobile_net.summary()
    
    return mobile_net