import numpy as np
import math


class NearestNeighbors:
    def __init__(self, data, label, k=1):
        self.x = data
        self.y = label
        self.k = k

    @staticmethod
    def euclid(dataSetSample, NewSample):
        return np.sqrt(np.sum((np.array(dataSetSample) - np.array(NewSample)) ** 2))

    @staticmethod
    def manhattan(dataSetSample, NewSample):
        return np.sqrt(np.sum(abs(np.array(dataSetSample) - np.array(NewSample))))

    def predict(self, newX):
        result = []
        for x, y in zip(self.x, self.y):
            cal = self.manhattan(x, newX)
            result.append([cal, y])
        result = sorted(result)[:self.k]
        countGroup = [0, 0]
        for item in result[:self.k]:
            if item[1] == 0:
                countGroup[0] += 1
            else:
                countGroup[1] += 1
        return countGroup.index(max(countGroup))


if __name__ == '__main__':
    DataSetData = [[1, 1], [1, 3], [2, 1], [2, 5], [3, 1], [4, 3], [4, 5], [5, 1], [5, 3], [6, 3], [6, 5]]
    DataSetLabel = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    N = NearestNeighbors(DataSetData, DataSetLabel, k=1)
    print(N.predict([3, 3]))
