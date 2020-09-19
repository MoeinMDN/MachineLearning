import numpy as np


class SingleVariableRegression:
    def __init__(self, data, label, learningRate=0.01):
        self.x = data
        self.y = label
        self.a = 1
        self.b = 1
        self.lr = learningRate
        self.predictResult = []

    def predict(self):
        self.predictResult = self.a * self.x + self.b
        return self.predictResult

    def updateParameter(self):
        inGradient = self.y - self.predict()
        N = len(self.y)
        self.b = self.b - self.lr * sum((-2 * inGradient)) / N
        self.a = self.a - self.lr * sum((-2 * x * inGradient)) / N
        return [self.a, self.b]

    def train(self, epoch):
        for _ in range(epoch):
            print('loss: ',sum((y - self.predict())**2) / len(self.y))
            self.updateParameter()
        return [self.a, self.b]


if __name__ == '__main__':
    x = np.array([1, 2, 2, 3])
    y = np.array([1, 1, 2, 2])
    reg = SingleVariableRegression(x, y, learningRate=0.05).train(900)
    print(reg)
