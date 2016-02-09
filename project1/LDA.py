import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.linalg import eig

"""
QDA and LDA for Stat 557

"""

class DA(object):
    """
    Linear Discriminant Analysis
    """
    import numpy as np

    def __init__(self, x, y):
        """
        inputs:
            x: matrix of predictor
            y: matrix of response
        """
        self.x = x
        self.y = y
        self.xDim = np.shape(x)
        self.yDim = np.shape(y)
        self.classSize = len(set(y))
        self.classMember = set(y)

    def preprocessing(self):
        """
        compute the mean and covariance matrix and class prior probability
        """
        self.classMean = {}
        self.priorProbability = {}

        for K in self.classMember:

            classK_data = self.x[self.y == K,]
            self.classMean[K] = np.transpose(np.mean(classK_data,0))

            self.priorProbability[K] = sum(self.y == K) / len(self.y)

        self.commonCovMatrix = np.cov(np.transpose(self.x))





    def LDA(self, x):
        delta_k = {}
        for K in self.classMember:
            delta_k[K] = np.dot(np.dot(x, self.commonCovMatrix),
                    self.classMean[K]) - .5 * np.dot(
                            np.dot(np.transpose(self.classMean[K]),
                                self.commonCovMatrix),
                   self.classMean[K]) + np.log(self.priorProbability[K])
        return max(delta_k, key = delta_k.get)

    def QDA():
        pass

    def fit(self, mode):
        self.y_hat = []
        if mode == "LDA":
            for X in self.x:
                self.y_hat.append(self.LDA(X))
        else:
            pass
        return self.y_hat

    def summary(self):
        self.accuracy = sum(self.y_hat == self.y) / len(self.y)
        print("accuracy: ", self.accuracy)



if __name__ == "__main__":
    np.random.seed(1234)
    cov = [[1.79, -0.14],[-.14,1.66]]
    mean1 = [0,0]
    mean2 = [2,2]
    Y = np.random.randint(0, 2, 1000)
    X1 = np.random.multivariate_normal(mean1, cov, 475)
    X2 = np.random.multivariate_normal(mean2, cov, 525)
    X = np.zeros((1000,2))
    X[Y==0] = X1
    X[Y==1] = X2
    print(Y)
    fit = DA(X, Y)
    print(fit.xDim, fit.yDim)
    fit.preprocessing()
    print(fit.classMean)
    print(fit.priorProbability)
    print(fit.commonCovMatrix)
    print(np.shape(fit.commonCovMatrix))
    print(fit.fit("LDA"))
    fit.summary()
