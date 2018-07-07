import numpy as np
import keras.backend as K

# mean score
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

# standard deviation
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

# earth movers loss
def emd_loss(y_true, y_pred):
    '''
    Earth Mover's Distance loss
    '''
    cdf_p    = K.cumsum(y_true, axis = -1)
    cdf_phat = K.cumsum(y_pred, axis = -1)
    loss     = K.mean(K.sqrt(K.mean(K.square(K.abs(cdf_p - cdf_phat)), axis = -1)))
    
    return loss