import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.centers = [list(np.random.randint(np.min(x), np.max(x), 2)) for _ in range(k)]
        self.category = self.initializeCategory()

    def initializeCategory(self):
        self.category = {}
        for i in range(self.k):
            self.category[f'{i}'] = []
        return self.category

    @staticmethod
    def euclidean(sample, center):
        return np.sqrt(np.sum((sample - center) ** 2))

    def categorizeDataWithCurrentCenter(self):
        for sample in self.x:
            distanceFromEachCenter = [self.euclidean(sample, self.centers[d]) for d in range(len(self.centers))]
            nameCategory = np.argmin(distanceFromEachCenter, axis=0)
            self.category[f'{nameCategory}'].append(sample)
        return self.category

    @staticmethod
    def findMeanOfCategoryItem(catContent):
        sumX = 0
        sumY = 0
        lengthCat = len(catContent)
        if lengthCat is 0:
            return 0
        for x, y, in catContent:
            sumX += x
            sumY += y
        return [sumX / lengthCat, sumY / lengthCat]

    def updateCenters(self):
        for cat in self.category.items():
            catContent = cat[1]
            catName = int(cat[0])
            meanCat = self.findMeanOfCategoryItem(catContent)
            if meanCat is not 0:
                self.centers[catName] = meanCat
        return self.centers

    def train(self, epoch):
        for _ in range(epoch):
            self.categorizeDataWithCurrentCenter()
            self.updateCenters()
        return self.centers


if __name__ == '__main__':
    data = np.array([
        [1, 1], [0, 0], [4, 4], [3, 3],
        [10, 10], [15, 15], [13, 13], [12, 12],
        [50, 50], [55, 55], [40, 40], [45, 45],
        [48, 48]])
    km = KMeans(data, k=2)
    km.train(100)
    centers = np.array(km.centers)
    plt.scatter(data[:, 0], data[:, 1], c='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='b')
    plt.show()