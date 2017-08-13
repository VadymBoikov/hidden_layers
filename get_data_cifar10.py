import keras
from keras.datasets import cifar10
import numpy as np
import pickle




num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

obj = {"x_train" :x_train, "y_train" :y_train,
       "x_test" :x_test, "y_test" :y_test}
with open("./tmp/data/cifar10.pickle", 'wb') as f:
    pickle.dump(obj, f)
