from keras.callbacks import Callback
import matplotlib
matplotlib.use('agg') # avoid _tkinter module not found
import matplotlib.pyplot as plt
from IPython.display import clear_output

class PlotLogs(Callback):
    
    def __init__(self, model_filename):
        self.model_filename = model_filename[:-3]
    
    def on_train_begin(self, logs = {}):
        self.x              = []
        self.losses         = []
        self.val_losses     = []
        self.fig            = plt.figure()
        self.logs           = []

    def on_epoch_end(self, epoch, logs = {}):
        
        # store the logs and current epoch
        self.logs.append(logs)
        self.x.append(epoch + 1)
        self.losses.append(self.logs[epoch]['loss'])
        self.val_losses.append(self.logs[epoch]['val_loss'])
        
        # clear the current output and print the plots
        clear_output(wait = True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.grid()
        plt.legend()
        plt.show()
        
    def on_train_end(self, logs={}):
        # save the logs
        with open('{}_score.txt'.format(self.model_filename), 'w') as f:
            print('train_losses: {}\n'.format(self.losses), file=f)
            print('val_losses: {}\n'.format(self.val_losses), file=f)
        
        plt.savefig('{}.png'.format(self.model_filename))
        
        print('Training finished!')