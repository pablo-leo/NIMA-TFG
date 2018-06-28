from keras.callbacks import Callback
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output

class PlotLogs(Callback):
    
    def on_train_begin(self, logs = {}):
        self.x          = []
        self.losses     = []
        self.val_losses = []
        self.fig        = plt.figure()
        self.logs       = []

    def on_epoch_end(self, epoch, logs = {}):
        
        # store the logs and current epoch
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(self.logs[epoch]['loss'])
        self.val_losses.append(self.logs[epoch]['val_loss'])
        
        # clear the current output and print the plots
        clear_output(wait = True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.grid()
        plt.legend()
        plt.show()