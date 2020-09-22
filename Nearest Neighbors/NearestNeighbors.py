import numpy as np
import math


class NearestNeighbors:
    def __init__(self, data, label):
        self.x = data
        self.y = label

    @staticmethod
    def euclid(dataSetSample, NewSample):
        return math.sqrt((dataSetSample[0] - NewSample[0]) ** 2 + (dataSetSample[1] - NewSample[1]) ** 2)

    @staticmethod
    def manhattan(dataSetSample, NewSample):
        return math.sqrt(abs(dataSetSample[0] - NewSample[0]) + abs(dataSetSample[1] - NewSample[1]))

    def predict(self, newX):
        result = []
        for x, y in zip(self.x, self.y):
            cal = self.manhattan(x, newX)
            result.append([cal, x, y])
        return sorted(result)


if __name__ == '__main__':
    DataSetData = [[1, 1], [1, 3], [2, 1], [2, 5], [3, 1], [4, 3], [4, 5], [5, 1], [5, 3], [6, 3], [6, 5]]
    DataSetLabel = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    N = NearestNeighbors(DataSetData, DataSetLabel)
    print(N.predict([3, 3]))
