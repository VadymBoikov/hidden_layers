"""

python train.py --DATA_PICKLED=tmp/data/mnist_w_corrup.pickle --SAVE_CHECKPOINT=tmp/checkpoints/mnist_article
"""
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PICKLED", help="train and test dataset", type=str)
parser.add_argument("--SAVE_CHECKPOINT", help="common storage save path",  type=str)
args = parser.parse_args()

checkpoint_path = args.SAVE_CHECKPOINT # tmp/checkpoints/checkpoint_article
data_pickled = args.DATA_PICKLED

batch_size = 64
num_classes = 10
epochs = 20

with open(data_pickled, 'rb') as f:
    obj = pickle.load(f)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']
    #y_train_corrup = obj['y_train_corrup']
    #y_test_corrup = obj['y_test_corrup']

input_shape = x_train.shape[1:]


def create_net():
    model = Sequential()

    # in tensorflow here is 64 filters, while in the article 32!
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape,
                     padding = 'same',
                     name='conv1'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', name='pool1'))
    model.add(BatchNormalization(name = 'normalization1'))

    model.add(Conv2D(64, (5, 5), activation='relu', padding= 'same', name = 'conv2'))
    model.add(BatchNormalization(name = 'normalization2'))

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


model = create_net()
checkpointer = ModelCheckpoint(filepath='%s.{epoch:02d}' % checkpoint_path, verbose=1)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[checkpointer])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
