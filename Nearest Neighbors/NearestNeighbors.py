from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import random


class NearestNeighbors:
    def __init__(self, data, k=1):
        self.x = []
        self.y = []
        for i in data:
            self.x.append(i[:-1])
            self.y.append(i[-1])
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
            if item[1] == 0:
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
    Test Algorithm In Cancer DataSet From Sklearn
    '''

    data = load_breast_cancer()
    DataSetData = data['data']
    DataSetLabel = data['target']

    DataSetData = pd.DataFrame(DataSetData)
    DataSetLabel = pd.DataFrame(DataSetLabel)

    allDataSet = pd.concat([DataSetData, DataSetLabel], axis=1).astype(float).values.tolist()
    random.shuffle(allDataSet)

    percentSlice = round(len(allDataSet) * 20 / 100)

    train, test = allDataSet[:percentSlice], allDataSet[percentSlice:]

    N = NearestNeighbors(train, k=3)
    total = 0
    correct = 0
    for item in test:
        pr = N.predict(item[:-1])
        if pr == item[-1]:
            correct += 1
        total += 1
    print('accuracy:', correct / total)
