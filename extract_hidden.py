"""


python extract_hidden.py --SAVE_DIR=tmp/hidden_layers/mnist_check00_10k --CHECKPOINT=tmp/checkpoints/mnist_article.00 --DATA_PICKLED=tmp/data/mnist_w_corrup.pickle

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

checkpoint_path = args.CHECKPOINT #checkpoint_path = "tmp/checkpoints/checkpoint_article.00"
save_dir = args.SAVE_DIR #save_dir = "tmp/layers_features"
data_pickled = args.DATA_PICKLED #"./tmp/data/mnist_w_corrup.pickle"

# save_dir = 'tmp/data/mnist_check00'
# >>> checkpoint_path = 'tmp/checkpoints/checkpoint_article.00'
# >>> data_pickled = 'tmp/data/mnist_w_corrup.pickle'

with open(data_pickled, 'rb') as f:
    obj = pickle.load(f)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']


model = load_model(checkpoint_path )
score = model.evaluate(x_test, y_test, verbose=0)
print("evaluation score is %s" %score)

layers_all = dict([(layer.name, layer.output) for layer in model.layers])

#hidden_layers_nms = ['conv2d_1', 'conv2d_2', 'dense_1', 'dense_2']
hidden_layers_nms = ['conv1', 'conv2', 'local3', 'local4']
extract_layers = {key: layers_all[key] for key in hidden_layers_nms}

print("starting evaluation")
layers_flat = []
for layer_name, layer_tensor in extract_layers.items():
    eval_function = K.function(inputs = [model.input, K.learning_phase()], outputs = [layer_tensor])

    #TODO change x_test to x_train !!!!!
    layer_out = eval_function([x_test, 0])[0]

    dim2 = np.prod(np.shape(layer_out)[1:])
    print("dimension of layer BEFORE %s is : %s" %(layer_name, np.shape(layer_out)))
    layer_flattened = np.reshape(layer_out , [-1, dim2])
    print("dimension of layer AFTER %s is : %s" %(layer_name, np.shape(layer_flattened)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt("%s/%s.csv" % (save_dir, layer_name), layer_flattened, delimiter = ',', fmt='%10.5e')

dim = np.prod(np.shape(x_test)[1:])
input_flattened = np.reshape(x_test, [-1, dim])
np.savetxt("%s/input.csv" % save_dir, input_flattened, delimiter = ',', fmt='%i')

np.savetxt("%s/trainingLabel.csv" % save_dir, y_test, delimiter = ',', fmt='%i')





