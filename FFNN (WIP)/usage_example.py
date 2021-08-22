import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

from ffnn import FFNN
from layers import Dense
from activations import Sigmoid, Softmax, ReLu
from costs import CategoricalCrossentropy
from utils import get_numeric_gradient_differences

def run_test():
    N_CLASS = 100

    X1 = np.random.randn(N_CLASS, 2) + np.array([0, -2])
    X2 = np.random.randn(N_CLASS, 2) + np.array([2, 2])
    X3 = np.random.randn(N_CLASS, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0] * N_CLASS + [1]*N_CLASS + [2]*N_CLASS).reshape(-1, 1)

    T = np.zeros((len(Y), 3))
    for i in range(len(Y)):
        T[i, Y[i]] = 1

    nn = FFNN( layers= [
        Dense(size=2, n_inputs=2, activation=ReLu), 
        Dense(size=3, n_inputs=2, activation=Sigmoid)#Softmax)
    ], cost = CategoricalCrossentropy)
    
    J_Hist = nn.train(X, T, epochs=50000, learning_rate=0.00003, convergence_treshhold=None)#1e-20)
    answer = np.argmax(nn(X), axis=1)

    ax1 = plt.subplot(221)
    ax1.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    ax1.set_title("Truth")

    ax2 = plt.subplot(222)
    ax2.scatter(X[:,0], X[:,1], c=answer, s=100, alpha=0.5)
    ax2.set_title("Prediction")

    ax3 = plt.subplot(212)
    ax3.plot(J_Hist)
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Cost")

    plt.show()

    # weight_differences, bias_differences, weight_info, bias_info = get_numeric_gradient_differences(nn, X, T, epsilon=1e-7)
    # ic(np.mean( np.abs(weight_differences) ))
    # ic(np.mean( np.abs(bias_differences) ))
    # ic(weight_differences, bias_differences)
    # ic(list(weight_info["backprop"] / (np.array(weight_info["backprop"]) + np.array(weight_info["numeric"]))))
    # ic(weight_info["numeric"], weight_info["backprop"])
    # ic(bias_info["numeric"], bias_info["backprop"])

    # ax = plt.subplot(111)
    # ax.scatter(y=weight_info["numeric"], x=range(len(weight_info["numeric"])), c="red")
    # ax.scatter(y=weight_info["backprop"], x=range(len(weight_info["backprop"])), c="blue")
    # ax.set_title("Red = numeric, Blue = backprop")
    # plt.show()

if __name__ == "__main__":
    run_test()