"""


python extract_hidden.py --DATA_PICKLED=tmp/data/mnist_w_corrup.pickle --SAVE_DIR=tmp/hidden_layers/features/model_mnist_ch19 --CHECKPOINT=tmp/checkpoints/model_mnist.00

"""

import keras
from keras import backend as K
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import pickle
import argparse
import os
num_classes = 10
# the data, shuffled and split between train and test sets


parser = argparse.ArgumentParser()
parser.add_argument("--SAVE_DIR", help="common storage save path", default= "tmp/layers_features/", type=str)
parser.add_argument("--DATA_PICKLED", help="train and test dataset", type=str)
parser.add_argument("--CHECKPOINT", help="common storage save path",  type=str)

args = parser.parse_args()

checkpoint_path = args.CHECKPOINT
save_dir = args.SAVE_DIR
data_pickled = args.DATA_PICKLED

with open(data_pickled, 'rb') as f:
    obj = pickle.load(f)
    x_train = obj['x_train']
    y_train = obj['y_train']


model = load_model(checkpoint_path )
layers_all = dict([(layer.name, layer.output) for layer in model.layers])

#hidden_layers_nms = ['conv1', 'conv2', 'local3', 'local4']
hidden_layers_nms = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
extract_layers = {key: layers_all[key] for key in hidden_layers_nms}

print("extract hidden layers")
for layer_name, layer_tensor in extract_layers.items():
    eval_function = K.function(inputs = [model.input, K.learning_phase()], outputs = [layer_tensor])

    layer_out = eval_function([x_train, 0])[0]

    dim2 = np.prod(np.shape(layer_out)[1:])
    print("dimension of layer BEFORE %s is : %s" %(layer_name, np.shape(layer_out)))
    layer_flattened = np.reshape(layer_out , [-1, dim2])
    print("dimension of layer AFTER %s is : %s" %(layer_name, np.shape(layer_flattened)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt("%s/%s.csv" % (save_dir, layer_name), layer_flattened, delimiter = ',', fmt='%.8e')

print("extract input")
dim = np.prod(np.shape(x_train)[1:])
layer_flattened = np.reshape(x_train, [-1, dim])
np.savetxt("%s/f0.csv" % save_dir, layer_flattened , delimiter = ',', fmt='%.8e')

print("extract labels")
np.savetxt("%s/trainingLabel.csv" % save_dir, y_train, delimiter = ',', fmt='%i')





