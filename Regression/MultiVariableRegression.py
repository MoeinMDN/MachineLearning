import numpy as np
import pandas as pd


class MultiVariableRegression:
    def __init__(self, x, y, learningRate=0.01):
        self.x = x
        self.y = y
        self.alpha = [0 for i in range(len(self.x[0]))]
        self.beta = 2
        self.lr = learningRate
        self.N = len(y)
        self.predictResult = []

    def predict(self):
        self.predictResult = self.x * self.alpha + self.beta
        self.predictResult = [sum(i) for i in self.predictResult]
        return self.predictResult

    def updateParameter(self):
        inGradient = self.predictResult - self.y
        self.beta -= self.lr * (sum(inGradient)) / self.N
        for a in range(len(self.alpha)):
            w = [i[a] for i in self.x]
            self.alpha[a] -= self.lr * sum(w * inGradient) / self.N
        return [self.alpha, self.beta]

    def train(self, epoch):
        for _ in range(epoch):
            self.updateParameter()
            loss = sum((self.y - self.predictResult) ** 2) / self.N
            print(f"iteration:{_} loss:{loss}")
            self.predict()
        return [self.alpha, self.beta]


if __name__ == '__main__':
    data = np.array([[1, 2], [3, 4], [5, 6]])
    label = np.array([10, 15, 18])
    reg = MultiVariableRegression(data, label)
    reg.predict()
    print(reg.train(10000))
