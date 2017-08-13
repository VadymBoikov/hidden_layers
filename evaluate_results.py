"""

python evaluate_results.py --CHECKPOINT=tmp/checkpoints/mnist_article.00 --DATA_PICKLED=tmp/data/mnist_w_corrup.pickle

"""

from keras.models import load_model
import pickle
import argparse
# the data, shuffled and split between train and test sets


parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PICKLED", help="train and test dataset", type=str)
parser.add_argument("--CHECKPOINT", help="common storage save path",  type=str)
args = parser.parse_args()

checkpoint_path = args.CHECKPOINT #checkpoint_path = "tmp/checkpoints/checkpoint_article.00"
data_pickled = args.DATA_PICKLED #"./tmp/data/mnist_w_corrup.pickle"

with open(data_pickled, 'rb') as f:
    obj = pickle.load(f)
    x_train = obj['x_train']
    y_train = obj['y_train']
    x_test = obj['x_test']
    y_test = obj['y_test']


model = load_model(checkpoint_path)
score_test = model.evaluate(x_test, y_test, verbose=0)
print("test score is %s \n" % score_test)


score_train= model.evaluate(x_train, y_train, verbose=0)
print("train score is %s \n" % score_train)