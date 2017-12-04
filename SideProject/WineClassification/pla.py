import pandas
import math

class Perceptron(object):
    #iteration_limit: int
    #eta: float: learning rate [0:1]
    
    def __init__(self, eta = 0.01, iteration=10):
        self.eta = eta
        self.iteration_limit = iteration
    def training(self, X, y):
        self.weight = numpy.zeros(1 + X.shape[1])
        self.errors_list = []

        for _ in range(self.iteration_limit):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                # print(update)
                self.weight[1:] += update * xi
                self.weight[0] += update
                errors += int(update != 0.0)
            self.errors_list.append(errors)
        return self

    def predict(self, X):
        return numpy.where(self.sign(self.get_input(X)), 1, -1)

    def sign(self, value):
        # print(value, value >=0)
        return value >= 0.0

    def get_input(self, X):
        return numpy.dot(X, self.weight[1:]) + self.weight[0]

def normalize(dataframe):
    result = dataframe.copy()
    for feature_name in dataframe.columns[1:]:
        max_value = dataframe[feature_name].max()
        min_value = dataframe[feature_name].min()
        result[feature_name] = (dataframe[feature_name] - min_value) / (max_value - min_value)
    return result

# https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
data = pandas.read_csv('wine.data').sample(frac=1)
data = normalize(data)

import matplotlib.pyplot as plot
import numpy

y = data.iloc[:, 0].values
print(y)
y = numpy.where(y == 1, 1, -1)
print(y)
X = data.iloc[:, 1:].values
print(type(X))

ppn = Perceptron(eta = 0.1, iteration=10)
ppn.training(X, y)
plot.plot(range(0, len(ppn.errors_list)), ppn.errors_list, marker='.')

# print(ppn.weight)

plot.show()