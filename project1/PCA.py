"""
Principle Component Analysis
"""

from numpy import shape, mean, var, dot, abs, real, reshape
from scipy.linalg import eig


class PCA(object):

    """
    Principle Componenet Analysis
    """

    def __init__(self, x):
        """
        inputs:
            numpy array with nxp dimension
        """

        self.x = x
        self.dim = shape(x)
        self.centerd_x = x - mean(x, 0)
        self.means = mean(x, 0)
        self.vars = var(x, 0)

    def fit(self):
        """
        returns:
            sorted eigen values/vectors tuple
         """
        eigval, eigvec = eig(dot(self.centerd_x.T, self.centerd_x))
        eig_pairs = [(abs(eigval[i]), eigvec[:, i]) for i in range(len(eigval))]
        eig_pairs.sort(reverse=True)

        return eig_pairs

    def summary(self, n=None):

        pairs = self.fit()
        d = [real(i[0]) for i in pairs]
        self.d = reshape(d, (1, self.dim[1]))

        if n is None:
            n = self.dim[1]

        var_exp = (d / sum(d))
        return var_exp[:n]

    def transform(self, n=None):
        pairs = self.fit()
        W = [i[1] for i in pairs]
        if n is None:
            n = self.dim[1]
        W = reshape(W, (self.dim[1], self.dim[1]), order='C')
        self.W = W.T
        X_new = dot(self.centerd_x, W.T[:, :n])
        return X_new

    def decomp(self):
        return self.d, self.W


if __name__ == "__main__":

    from numpy.random import seed, multivariate_normal

    seed(1234)
    means = [0, 0]
    cov = [[2, 0], [0, 2]]
    data = multivariate_normal(means, cov, 20000)
    pca = PCA(data)
    pca.fit()
    print("summary \n ", pca.summary())
    pca.transform()
    print(pca.decomp())
