from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x1 = [i for i in range(100)]
    x2 = [i for i in range(100)]
    X = np.array(list(zip(x1, x2)))
    y = np.reshape([(i ** 2) + 100 for i in range(100)], (-1, 1))

    #X = (X - np.mean(X)) / np.std(X) #scaling input does wonders

    model = LinearRegression(n_features=X.shape[1], add_intercept=True)
    costs = model.fit(X=X, y=y, epochs=100000, learning_rate=0.0001)
    preds = model.predict(X)

    plt.plot(costs)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:,0], X[:,1], preds, c="red")
    ax.scatter(X[:,0], X[:,1], y, c="blue")
    plt.show()