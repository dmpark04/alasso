# Keras-specific functions and utils
from keras.callbacks import Callback
from keras.models import Model
import keras.backend as K
from keras.layers import Dense
import tensorflow as tf

class LossHistory(Callback):
    def __init__(self, *args, **kwargs):
        super(LossHistory, self).__init__(*args, **kwargs)
        self.losses = []
        self.regs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.regs.append(K.get_session().run(self.model.optimizer.regularizer))


