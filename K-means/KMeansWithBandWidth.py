import numpy as np
import matplotlib.pyplot as plt


class KMeansWithBandWidth:
    def __init__(self, x, k):
        self.x = x
        self.k = k


if __name__ == '__main__':
    data = np.array([
        [1, 1], [0, 0], [4, 4], [3, 3],
        [10, 10], [15, 15], [13, 13], [12, 12],
        [50, 50], [55, 55], [40, 40], [45, 45],
        [48, 48]])
    km = KMeansWithBandWidth(data, k=2)
    plt.scatter(data[:, 0], data[:, 1], c='k')
    plt.show()
