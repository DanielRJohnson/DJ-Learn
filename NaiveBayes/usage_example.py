from NBClassifier import GaussianNB
from pandas import read_csv
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    forest = GaussianNB()

    df = read_csv("iris.csv")
    df = df.values # pd.DataFrame -> np.array

    # Split on target
    X = df[:,0:4]
    y = df[:,4]

    forest.fit(X, y)
    print("Our Accuracy:", accuracy_score(y, forest.predict(X)))
