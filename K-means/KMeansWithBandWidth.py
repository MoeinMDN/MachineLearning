import numpy as np
import matplotlib.pyplot as plt


class KMeansWithBandWidth:
    def __init__(self, x, bandWidth):
        self.x = x
        self.r = bandWidth
        self.centers = x
        self.categorize = {}

    @staticmethod
    def uniqueNewCenters(centers):
        uniqueCenters = []
        for i in centers:
            if i not in uniqueCenters:
                uniqueCenters.append(i)
        return uniqueCenters

    @staticmethod
    def euclidean(sample, point):
        return np.linalg.norm(point - sample)

    def categorizeData(self):
        categorize = {}
        for center in self.centers:
            categorize[f'{center}'] = []
            tempCat = []
            for j in self.x:
                tempCat.append([self.euclidean(center, j), list(j)])
            tempCat = sorted(tempCat)
            categorize[f'{center}'].append(tempCat[:self.r])
        return categorize

    def updateCenters(self, categorize):
        tempCenters = []
        for cat in categorize.items():
            _, points = cat
            points = points[0]
            points = np.array([j[1] for j in points])
            points = np.ndarray.sum(points, axis=0) / len(points)
            tempCenters.append(list(tuple(points)))
        self.centers = self.uniqueNewCenters(tempCenters)
        return self.centers

    def train(self):
        while True:
            categorize = self.categorizeData()
            self.updateCenters(categorize)
            if categorize == self.categorize:
                break
            self.categorize = categorize
        return self.centers


if __name__ == '__main__':
    data = np.array([
        [1, 1], [0, 0], [4, 4], [3, 3],
        [10, 10], [15, 15], [13, 13], [12, 12],
        [50, 50], [55, 55], [40, 40], [45, 45],
        [48, 48]])
    km = KMeansWithBandWidth(data, bandWidth=5)
    km.train()
    centers = np.array(km.centers)
    print(centers, 'centers')
    plt.scatter(data[:, 0], data[:, 1], c='k')
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='b')
    plt.show()