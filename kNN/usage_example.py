from kNN import kNNClassifier
from pandas import read_csv
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    kNN = kNNClassifier()

    df = read_csv("iris.csv")
    df = df.values # pd.DataFrame -> np.array

    # Split on target
    X = df[:,0:4]
    y = df[:,4]

    kNN.fit(X, y, n_neighbors=5)
    print("Our Accuracy:", accuracy_score(y, kNN.predict(X)))
