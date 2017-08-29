from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Activation

## TODO rename layers name. and they should start with t_, tmp_, test_ !!!!
def net_mnist(num_classes, input_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation=None,
                     strides=(1, 1),
                     padding = 'same',
                     input_shape=input_shape,
                     kernel_initializer = 'TruncatedNormal',
                     name='f1')) # aplha for this layer
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same',
                           name = 'f2')) # aplha for this layer

    model.add(Conv2D(64, kernel_size=(5, 5),
                     activation=None,
                     strides=(1, 1),
                     padding = 'same',
                     input_shape=input_shape,
                     kernel_initializer = 'TruncatedNormal',
                     name='f3')) # aplha for this layer
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same',
                           name = 'f4')) # aplha for this layer
    model.add(Flatten())
    #todo check how regularizer work in wf estimation!
    model.add(Dense(512, activation=None,
                    kernel_initializer = 'TruncatedNormal',))
                    #kernel_regularizer=regularizers.l2(5e-4),
                    #bias_regularizer=regularizers.l2(5e-4))) #

    model.add(Activation('relu', name = 'f5')) #aplha for this layer
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=None, name = 'f6',
                    kernel_initializer = 'TruncatedNormal',))
                    #kernel_regularizer=regularizers.l2(5e-4),
                    #bias_regularizer=regularizers.l2(5e-4))) # aplha for this layer
    model.add(Activation('softmax'))

    #todo be sure about optimizer
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
