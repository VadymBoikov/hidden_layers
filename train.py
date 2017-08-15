"""
 python train.py --DATA_PICKLED=tmp/data/mnist_w_corrup.pickle --SAVE_CHECKPOINT=tmp/checkpoints/model_mnist
"""
from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import numpy as np
import pickle
import argparse
import nets

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PICKLED", help="train and test dataset", type=str)
parser.add_argument("--SAVE_CHECKPOINT", help="common storage save path",  type=str)
args = parser.parse_args()
CHECKPOINT_PATH = args.SAVE_CHECKPOINT # tmp/checkpoints/checkpoint_article
DATA_PICKLED = args.DATA_PICKLED
# CHECKPOINT_PATH = "tmp/checkpoints/test_checkpoint"
# DATA_PICKLED = "tmp/data/mnist_w_corrup.pickle"

BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 20

with open(DATA_PICKLED, 'rb') as f:
    obj = pickle.load(f)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']
    #y_train_corrup = obj['y_train_corrup']
    #y_test_corrup = obj['y_test_corrup']

input_shape = x_train.shape[1:]

model = nets.create_net_mnist(NUM_CLASSES, input_shape)
checkpointer = ModelCheckpoint(filepath='%s.{epoch:02d}' % CHECKPOINT_PATH, verbose=1)


initial_lr = 0.01
lrate=LearningRateScheduler(lambda epoch: initial_lr * 0.95 ** epoch )

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[checkpointer])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
