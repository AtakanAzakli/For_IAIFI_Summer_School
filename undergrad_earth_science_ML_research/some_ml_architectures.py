import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Conv2D, LSTM, GRU, Dense, Flatten,
                                     MaxPooling1D, MaxPool2D, SpatialDropout1D, SpatialDropout2D,
                                     Bidirectional, BatchNormalization, Activation, add)

class BaseModel(Model):
    """
    Base model class providing common utility functions.
    """
    def __init__(self, nb_filters, kernel_size, padding='same', activation='relu', drop_rate=0.1):
        super(BaseModel, self).__init__()
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.drop_rate = drop_rate

    def encoder(self, inpC, depth=7):
        """ Returns an encoder combining residual blocks and max pooling. """
        e = inpC
        for dp in range(depth):
            e = Conv1D(self.nb_filters[dp], self.kernel_size[dp], padding=self.padding, activation=self.activation)(e)
            e = MaxPooling1D(2, padding=self.padding)(e)
        return e

    def cnn_block(self, inpC, filters, kernel_size=3):
        """ Returns a CNN residual block. """
        prev = inpC
        x = BatchNormalization()(prev)
        x = Activation(self.activation)(x)
        x = SpatialDropout1D(self.drop_rate)(x, training=True)
        x = Conv1D(filters, kernel_size, padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = SpatialDropout1D(self.drop_rate)(x, training=True)
        x = Conv1D(filters, kernel_size, padding=self.padding)(x)
        return add([prev, x])

    def bilstm_block(self, inpR, filters):
        """ Returns a BiLSTM residual block. """
        x_rnn = Bidirectional(LSTM(filters, return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate))(inpR)
        x_rnn = Conv1D(filters, 1, padding=self.padding)(x_rnn)
        return BatchNormalization()(x_rnn)

    def bigru_block(self, inpR, filters):
        """ Returns a BiGRU residual block. """
        x_rnn = Bidirectional(GRU(filters, return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate))(inpR)
        x_rnn = Conv1D(filters, 1, padding=self.padding)(x_rnn)
        return BatchNormalization()(x_rnn)

class EQT_BiGRU(BaseModel):
    def __init__(self, nb_filters, kernel_size):
        super(EQT_BiGRU, self).__init__(nb_filters, kernel_size)
        self.flatten = Flatten()
        self.dense = Dense(2, activation="softmax")

    def call(self, inputs):
        x = self.encoder(inputs)
        for _ in range(5):
            x = self.cnn_block(x, self.nb_filters[-1])
        for _ in range(3):
            x = self.bigru_block(x, self.nb_filters[1])
        x = self.flatten(x)
        return self.dense(x)

class Residual_CNN(BaseModel):
    def __init__(self, nb_filters, kernel_size):
        super(Residual_CNN, self).__init__(nb_filters, kernel_size)
        self.flatten = Flatten()
        self.dense = Dense(2, activation="softmax")

    def call(self, inputs):
        x = Conv2D(64, 3, padding=self.padding, activation=self.activation)(inputs)
        for _ in range(5):
            x = self.cnn_block(x, 64, kernel_size=3)
        x = self.flatten(x)
        return self.dense(x)
