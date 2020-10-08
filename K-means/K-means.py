import numpy as np


class KMeans:
    def __init__(self, x, k):
        self.x = x
        self.k = k


if __name__ == '__main__':
    data = np.array([[1, 1], [5, 5]])
    svm = KMeans(data, k=2)
