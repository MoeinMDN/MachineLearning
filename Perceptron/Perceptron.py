import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, x, y, learningRate=0.01):
        self.x = x
        self.y = y
        self.addExtraColumnToXSample()
        self.w = np.zeros(len(self.x[0]))
        self.lr = learningRate

    def addExtraColumnToXSample(self):
        self.x = np.hstack((self.x, np.ones((self.x.shape[0], 1), dtype=self.x.dtype)))
        return self.x

    @staticmethod
    def checkSample(sample):
        if len(sample) == 2:
            sample = list(sample)
            sample.append(1)
        return np.array(sample)

    def predict(self, sample):
        sample = self.checkSample(sample)
        activation = np.dot(self.w, sample)
        return 1.0 if activation >= 0 else 0

    def train(self, epoch):
        for _ in range(epoch):
            for k, p in zip(self.x, self.y):
                pri = self.predict(sample=k)
                self.updateParameter(pri, p, k)

    def updateParameter(self, predict, trueLabel, xData):
        self.w += self.lr * (trueLabel - predict) * xData
        return self.w


if __name__ == '__main__':
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    label = np.array([1, 1, 1, 0])
    per = Perceptron(data, label)
    per.train(100)
    print(per.predict([0, 0]))
