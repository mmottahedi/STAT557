import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.linalg import eig


class DA(object):
    """
    Linear Discriminant Analysis
    """
    def __init__(self, x, y):
        """
        inputs:
            x: matrix of predictor
            y: matrix of response
        """
        self.x = sp.matrix(x)
        self.y = sp.matrix(y)
        self.xDim = np.shape(x)
        self.yDim = np.shape(y)
        self.classSize = len(set(y))
        self.class_member = set(y)

    def LDA():
        pass

    def QDA():
        pass

    def fit():
        pass

    def summary():
        pass



if __name__ == "main":

    x = np.random.rand(5000).reshape(1000,5)
    y = np.random.randint(0, 2, 1000)

    classifier = LDA(x,y)
    print(classifier.xDim, classifier.yDim)

