import numpy as np

class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.means = np.array([np.mean(X[np.where(y == c)], axis=0) for c in unique_classes])
        self.variances = np.array([np.var(X[np.where(y == c)], axis=0) for c in unique_classes])

        # simple probabilities (class_count / total_count)
        self.prior_class_probabilities = class_counts/len(y)
        # mapping from class_index back to whatever the real value is
        self.class_index_to_class = {index:unique_classes[index] for index in range(len(unique_classes))}

    def predict(self, X: np.ndarray) -> list:
        predictions = []
        for row in X:
            posterior_numerator = np.zeros(self.prior_class_probabilities.shape)
            for class_index in range(len(posterior_numerator)):
                prior = self.prior_class_probabilities[class_index]
                # likelyhood[class(i)][feature(f)] = exp( -(x_c_f - u_c_f)^2 / (2(var_c_f)) )
                likelyhood = np.array([
                    np.exp(
                            -np.square(row[feature_index] - self.means[class_index][feature_index]) 
                            / 
                            (2 * self.variances[class_index][feature_index])
                    )
                    for feature_index in range(len(row))
                ])
                # posterior_numerator = prior * likelyhood_c1_f1 ... likelyhood_cn_fn
                posterior_numerator[class_index] = np.product(np.concatenate([[prior], likelyhood]))
            predictions.append(self.class_index_to_class[posterior_numerator.argmax()])
        return predictions