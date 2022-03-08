import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, batch_size=32, is_training = True, weight = 1, shuffle=True):
        'Initialization'
        self.x = x
        self.y = self._data_generation(y)
        self.weights = self._get_weights(is_training,weight)
        self.length = len(x)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        return self.x[indexes], self.y[indexes], self.weights[indexes]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _data_generation(self, y):

        y_not = np.logical_not(y)

        return np.stack([y_not,y],axis=1)

    def _get_weights(self, is_training, weight):
        
        weights = np.ones(self.y.shape[0])

        if is_training:

            weights[self.y[:,1]==1] = weight

        return weights


