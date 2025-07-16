import numpy as np
import pandas as pd
def activationFunuction(xi, yi, w, b):
    #"prediction" with current weights, checks if prediction is correct,
    # sends back decision and penalty
    penalty =(1 - yi * (np.dot(w, xi) + b))
    #make a decision
    if penalty >= 0:
        return 1, penalty
    else:
        return 0, 0

def computeGradient(X, Y, w, b ):
    numofsamples, features = X.shape
    # deltaw and delta b reset back to zero after every iteration
    deltaw = np.array([0] * features)
    deltab = 0
    wrong = 0
    loss = 0

    for i in range(0, numofsamples):
        xi = X[i] #data, observations
        yi = Y[i] #labels

        decision, penalty = activationFunuction(xi, yi, w, b)
        if decision == 1:
            wrong = wrong +  1
            loss += (penalty **2 )
        deltab = deltab + ((-2) * yi * penalty * decision)
        deltaw = deltaw + ((-2) * yi * xi * penalty * decision)
        # divide by numofsamples  for numerical stability
    return deltaw / numofsamples, deltab / numofsamples , wrong, loss

"""
X is data and Y are the labels
alpha is the learning rate, step size if you will
"""

def subgradientDescent(X, y, num_iter, alpha):
    iterations = [1, 10, 100, 1000, 10000, 100000]
    numofsamples, features = X.shape

    ITERATIONS = num_iter

    w = np.array([0] * features)
    b = 0
    LEARNING_RATE = alpha
    x = 0

    while x < ITERATIONS:
        x += 1
        #compute gradient 
        deltaw, deltab, wrong, loss = computeGradient(X, y, w, b)

        # update
        w = w - (LEARNING_RATE) * deltaw
        b = b - (LEARNING_RATE) * deltab
        norm = np.dot(deltaw, deltaw) + (deltab * deltab)
        if x in iterations:
            print("Iteration:%d, Weight:%s, Bias:%s" % (x, w, b))
            print(f"Loss = {loss}")
            print(
                "Norm:%s, subgradient of w:%s, sub gradient of b:%s, misclassifications:%d" % (
                    norm, deltaw, deltab, wrong))
            print()


if __name__ == '__main__':
    # read data
    df = pd.read_csv('perceptron.data', header=None)

    features = df.iloc[:, 0:4].values
    labels = df.iloc[:, 4].values
    subgradientDescent(features, labels, 100002, 0.1)












