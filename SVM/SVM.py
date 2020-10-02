import numpy as np


class SVM:
    def __init__(self, x, y, learningRate=0.01):
        self.x = x
        self.lr = learningRate
        self.addExtraColumnToXSample()
        self.y = y
        self.regularization = 5
        self.w = np.ones(len(self.x[0]))
        self.loss = None

    def addExtraColumnToXSample(self):
        self.x = np.hstack((self.x, np.ones((self.x.shape[0], 1), dtype=self.x.dtype)))
        return self.x

    def lossOfTheParameter(self):
        N = len(self.y)
        calculateMax = 1 - self.y * (np.dot(self.x, self.w))
        calculateMax[calculateMax < 0] = 0
        hingeFormula = 1 - self.y * (np.dot(self.x, self.w))
        hingeFormula[hingeFormula < 0] = 0
        hingeFormula = self.regularization * sum(hingeFormula) / N
        self.loss = 2 * np.dot(self.w, self.w) + hingeFormula
        return [self.loss, self.loss / 100]

    def predict(self, samples):
        return [np.sign(np.dot(self.w, s)) for s in samples]

    def calculateCostGradient(self):
        # original code of this function :
        # https://github.com/qandeelabbassi/python-svm-sgd/blob/master/svm.py#L59
        hingeFormula = 1 - self.y * (np.dot(self.x, self.w))
        dw = np.zeros(len(self.w))

        for ind, d in enumerate(hingeFormula):
            if max(0, d) == 0:
                di = self.w
            else:
                di = self.w - (self.regularization * self.y[ind] * self.x[ind])
            dw += di
        dw = dw / len(self.y)  # average
        return dw

    def updateWeight(self):
        self.w -= self.lr * (self.calculateCostGradient())
        return self.w

    def train(self, epoch):
        for _ in range(epoch):
            pureLoss, percentLoss = self.lossOfTheParameter()
            self.updateWeight()
            print(f'epoch:{_} loss:{percentLoss} weight:{self.w}')
        return self.w


if __name__ == '__main__':
    data = np.array([[1, 1], [5, 5]])
    label = np.array([-1, 1])
    svm = SVM(data, label, learningRate=0.000001)
    svm.train(100000)
