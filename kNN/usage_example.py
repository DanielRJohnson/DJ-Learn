from kNNClassifier import kNNClassifier
from kNNRegressor import kNNRegressor
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    kNNCl = kNNClassifier()
    kNNRg = kNNRegressor()

    df = read_csv("iris.csv")
    df = df.values # pd.DataFrame -> np.array

    # Split on target
    X = df[:,0:4]
    y = df[:,4]

    kNNCl.fit(X, y, n_neighbors=5)
    print("Our Accuracy:", accuracy_score(y, kNNCl.predict(X)))

    regressionX = df[:,0:3]
    regressionY = df[:,3]
    kNNRg.fit(regressionX, regressionY, n_neighbors=5)
    print("Our regression error:", mean_squared_error(regressionY, kNNRg.predict(regressionX)))