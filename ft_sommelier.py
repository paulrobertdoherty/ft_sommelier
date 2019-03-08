import os
import math
import random
import pandas
import matplotlib.pyplot as pyplot

data = pandas.read_csv("./resources/winequality-red.csv", sep=';')

class Perceptron:
    def __init__(self, inputs, seed=None):
        self.inputs = inputs
        if seed != None:
            random.seed(seed=seed)
        else:
            random.seed(seed=random.getrandbits(128));
        self.bias = random.random();
        self.weights = []
        for i in range(inputs):
            self.weights.append(random.random())

    def copy_weights(self, weights):
        weights_copy = []
        for weight in weights:
            weights_copy.append(weight)
        return weights_copy

    @staticmethod
    def predict_with(input_vals, wab, step_activation=False):
        output = 0
        for i in range(len(input_vals)):
            output += input_vals[i] * wab[0][i]
        tr = output * wab[1]
        if not step_activation:
            return tr
        if tr > 0:
            return 1
        return 0

    def predict(self, input_vals, step_activation=False, possibility=None):
        if possibility == None:
            return predict_with(input_vals, (self.weights, self.bias), step_activation=step_activation)
        return predict_with(input_vals, possibility, step_activation=step_activation)

    def get_all_weights(self, learning_rate):
        weights = []


    def get_possibilties(self, learning_rate):
        possibilities = []
        weights = get_all_weights(learning_rate, 0)
        for weight in weights:
            for i in range(2):
                possibility = []
                if i % 2 == 0:
                    possibility[1] = bias - learning_rate
                else:
                    possibility[1] = bias + learning_rate
                possibility[0] = weight
                possibilities.append(possibility)

    def train_epoch(self, training_data, learning_rate):
        lowest_error = 0
        lowest_index = -1
        possibilities = get_possibilities(learning_rate)
        errors = []
        for possibility in possibilities:
            error = 0
            for i in range(training_data.shape[1]):
                error += predict(vals, step_activation=True, possibility=possibility) #TODO: subtract something from here
            if lowest_index == -1 or lowest_error > error:
                lowest_index = i
                lowest_error = error
            errors.append(error)
        self.weights = possiblities[lowest_index][0]
        self.bias = possiblities[lowest_index][1]
        return errors[lowest_index]

    def train(self, training_data, epochs, learning_rate):
        if epochs <= 0:
            while True:
                error = train_epoch
                if error == 0:
                    return (0, 0, self.weights, self.bias)
        tr = []
        for epoch in epochs:
            error = train_epoch(training_data, learning_rate)
            tr.append((epoch, error, copy_weights(self.weights), self.bias))
            if error == 0:
                break
        return tr