from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import random


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

    def analyzeResult(self, predictResult):
        result = sorted(predictResult)[:self.k]
        countGroup = [0, 0]
        for item in result[:self.k]:
            if item[1][0] == 0:
                countGroup[0] += 1
            else:
                countGroup[1] += 1
        return countGroup.index(max(countGroup))

    def predict(self, newX):
        result = []
        for x, y in zip(self.x, self.y):
            cal = self.euclid(x, newX)
            result.append([cal, y])
        return self.analyzeResult(result)


if __name__ == '__main__':

    '''
    Test Algorithm In Cancer Data Set
    '''

    data = load_breast_cancer()
    DataSetData = data['data']
    DataSetLabel = data['target']

    DataSetData = pd.DataFrame(DataSetData)
    DataSetLabel = pd.DataFrame(DataSetLabel)

    DataSetData = DataSetData.astype(float).values.tolist()
    DataSetLabel = DataSetLabel.astype(float).values.tolist()

    N = NearestNeighbors(DataSetData, DataSetLabel, k=3)
    total = 0
    correct = 0
    for item in DataSetData:
        pr = N.predict(item)
        # print(pr, DataSetLabel[total][0])
        if pr == DataSetLabel[total][0]:
            correct += 1
        total += 1
    print('accuracy:', correct / total)
