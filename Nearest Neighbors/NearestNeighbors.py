from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import random


class NearestNeighbors:
    def __init__(self, data, k=1, measure='e'):
        self.x = []
        self.y = []
        self.data = data
        validMeasure = ['e', 'm', 'manhattan', 'euclid']
        if measure in validMeasure:
            self.measure = measure
        else:
            raise IOError(f'use one of the valid measure: {validMeasure}')
        self.initialize()
        self.LengthOfLabels = self.findLengthOfLabels()
        self.k = k

    def initialize(self):
        for i in self.data:
            self.x.append(i[:-1])
            self.y.append(i[-1])
        return [self.x, self.y]

    @staticmethod
    def euclid(dataSetSample, NewSample):
        return np.sqrt(np.sum((np.array(dataSetSample) - np.array(NewSample)) ** 2))

    @staticmethod
    def manhattan(dataSetSample, NewSample):
        return np.sum(abs(np.array(dataSetSample) - np.array(NewSample)))

    def findLengthOfLabels(self):
        temp = np.unique(self.y)
        return {'groupName': list(temp), 'length': len(temp)}

    def analyzeResult(self, predictResult):
        result = sorted(predictResult)[:self.k]
        countGroup = list(np.zeros(self.LengthOfLabels['length']))
        for item in result[:self.k]:
            countGroup[self.LengthOfLabels['groupName'].index(item[1])] = +1
        return countGroup.index(max(countGroup))

    def predict(self, newX):
        result = []
        cal = 0
        for x, y in zip(self.x, self.y):
            if self.measure in ['e', 'euclid']:
                cal = self.euclid(x, newX)
            if self.measure in ['m', 'manhattan']:
                cal = self.manhattan(x, newX)
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

    N = NearestNeighbors(train, k=3, measure='m')

    total = 0
    correct = 0
    for item in test:
        pr = N.predict(item[:-1])
        if pr == item[-1]:
            correct += 1
        total += 1
    print('accuracy:', correct / total)
