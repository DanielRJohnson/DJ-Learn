from LogisticRegression import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N_SAMPLES_PER_CLASS = 100
    g_cloud_1 = np.random.randn(N_SAMPLES_PER_CLASS, 2) + np.array([10, 10])
    g_cloud_2 = np.random.randn(N_SAMPLES_PER_CLASS, 2) + np.array([20, 20])
    X = np.vstack([g_cloud_1, g_cloud_2])

    y = np.array([0]*N_SAMPLES_PER_CLASS + [1]*N_SAMPLES_PER_CLASS).reshape(-1, 1)

    plt.scatter(X[:,0], X[:,1], c=y, s=100, alpha=0.5)
    plt.axis("equal")
    plt.show()

    model = LogisticRegression(n_features=X.shape[1], add_intercept=True)

    likelihoods = model.fit(X=X, y=y, epochs=100000, learning_rate=0.0001)
    plt.plot(likelihoods)
    plt.show()

    preds = model.predict(X)
    plt.scatter(X[:,0], X[:,1], c=preds, s=100, alpha=0.5)
    plt.axis("equal")
    plt.show()