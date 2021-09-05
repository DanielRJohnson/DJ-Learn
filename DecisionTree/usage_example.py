from RandomForestClassifier import RandomForestClassifier
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SklearnRFClassifier

if __name__ == "__main__":
    forest = RandomForestClassifier(forest_size=100, max_depth=3)
    theirForest = SklearnRFClassifier(n_estimators=100, max_depth=3)

    df = read_csv("iris.csv")
    df = df.values # pd.DataFrame -> np.array

    # Split on target
    X = df[:,0:4]
    y = df[:,4]

    forest.fit(X, y, rows_subset_fraction = 0.15, pbar=True)
    theirForest.fit(X, y)

    print("Accuracy will depend greatly on forest_size, max_depth, and rows_subset_fraction")
    print("Our Accuracy:", accuracy_score(y, forest.predict(X)))
    print("SkLearn Accuracy:", accuracy_score(y, theirForest.predict(X)))