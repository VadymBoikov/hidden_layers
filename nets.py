
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
from keras import regularizers

## TODO rename layers name. and they should start with t_, tmp_, test_ !!!!
def create_net_mnist(num_classes, input_shape):
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
    model.add(Dense(512, activation=None, name = 'f5',
                    kernel_initializer = 'TruncatedNormal',
                    kernel_regularizer=regularizers.l2(5e-4),
                    bias_regularizer=regularizers.l2(5e-4))) # aplha for this layer

    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=None, name = 'f6',
                    kernel_initializer = 'TruncatedNormal',
                    kernel_regularizer=regularizers.l2(5e-4),
                    bias_regularizer=regularizers.l2(5e-4))) # aplha for this layer
    model.add(Activation('softmax'))

    #todo be sure about optimizer
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(momentum=0.9), #AdaDelta
                  metrics=['accuracy'])
    return model


def create_net_article(num_classes, input_shape):
    model = Sequential()

    # in tensorflow here is 64 filters, while in the article 32!
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape,
                     padding = 'same',
                     name='conv1'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', name='pool1'))
    ### here add Local response normalization (not implmented)
    model.add(BatchNormalization(name = 'normalization1'))

    model.add(Conv2D(64, (5, 5), activation='relu', padding= 'same', name = 'conv2'))
    model.add(BatchNormalization(name = 'normalization2'))
    ## here add another LRN !

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', name='pool2'))

    # no dropout here!
    model.add(Flatten())
    model.add(Dense(384, activation = 'relu', name='local3'))
    model.add(Dense(192, activation = 'relu', name='local4'))

    model.add(Dense(num_classes, activation='softmax'))

    # maybe add learning rate decay
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])
    return model
