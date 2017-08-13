import keras
from keras.datasets import mnist
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--SAVE_PATH", default = "tmp/data/mnist_w_corrup.pickle", type=str)

args = parser.parse_args()
save_path = args.SAVE_PATH



def corrupt_label(label, perc, num_classes):
    if np.random.random(1) < perc:
        choose_from = np.delete(range(num_classes), label)
        return np.random.choice(choose_from, 1)[0]
    else:
        return label


def make_corrupted_labels(y_labels, corrup_levels, num_classes):
    list_corruptions = []
    for corrup_level in corrup_levels:
        corrupted_y = []
        for label in y_labels:
            corrupted_y.append(corrupt_label(label, corrup_level, num_classes))
        corrupted_y = keras.utils.to_categorical(corrupted_y, num_classes)
        list_corruptions.append(corrupted_y)
    return list_corruptions


num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train_corrup = make_corrupted_labels(y_train, [0.25, 0.5, 0.75, 1], num_classes)
y_test_corrup = make_corrupted_labels(y_test, [0.25, 0.5, 0.75, 1], num_classes)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

obj = {"x_train" :x_train, "y_train" :y_train,
       "x_test" :x_test, "y_test" :y_test,
       "y_train_corrup" :y_train_corrup, "y_test_corrup" : y_test_corrup}
with open(save_path, 'wb') as f:
    pickle.dump(obj, f)
