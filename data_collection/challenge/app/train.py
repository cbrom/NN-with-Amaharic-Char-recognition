import numpy as np
from sklearn.neural_network import MLPClassifier

def load_dataset():
    target = np.load('datasets/geez-numeral-target.npz', 'r')
    data = np.load('datasets/geez-numeral-data.npz', 'r')

    return target, data


if __name__ == '__main__':

    target, data = load_dataset()

