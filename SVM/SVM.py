import numpy as np


class SVM:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = np.zeros(len(self.x[0]))

    def train(self, epochs):
        eta = 1
        for epoch in range(1, epochs):
            for i, x in enumerate(self.x):
                if (self.y[i] * np.dot(self.x[i], self.w)) < 1:
                    self.w = self.w + eta * ((self.x[i] * self.y[i]) + (-2 * (1 / epoch) * self.w))
                else:
                    self.w = self.w + eta * (-2 * (1 / epoch) * self.w)
        return self.w

    def predict(self, Samples):
        lastIndex = len(self.w) - 1
        result = []
        for Sample in Samples:
            cal = np.sign(sum(Sample * self.w[:lastIndex - 1]) - self.w[lastIndex])
            result.append(cal)
        return result


if __name__ == '__main__':
    data = np.array([[1, 1, -1], [2, 2, -1]])
    label = np.array([-1, 1])
    svm = SVM(data, label)
    svm.train(100000)
    print(svm.w)
    print(svm.predict([[1, 1]]))
