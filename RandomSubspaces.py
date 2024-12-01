import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from UniformSubsampling import UniformSubsampler
import random

class RandomSubspacesMethod:
    def __init__(self, n_estimators=100, random_state=42, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.random_state = random_state
        random.seed = self.random_state
        np.random.seed = self.random_state
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []  # Store feature indices for each tree


    def fit(self, X, y):
        _, n_features = X.shape
        if self.max_features == 'sqrt':
            n_features_to_use = int(np.sqrt(n_features))
        else:
            n_features_to_use = self.max_features  # or any other criteria
        self.feature_indices = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier()
            selected_features = np.random.choice(range(n_features), n_features_to_use, replace=False)
            tree.fit(X[:, selected_features], y)
            self.trees.append(tree)
            self.feature_indices.append(selected_features)


    def predict(self, X):
        predictions = np.array([tree.predict(X[:, features_idx]) for tree, features_idx in zip(self.trees, self.feature_indices)])
        final_predictions = stats.mode(predictions, axis=0).mode
        return np.squeeze(final_predictions)
    

    def get_params(self, deep=True):
        # Return a dictionary of parameters, similar to scikit-learn's get_params
        return {"n_estimators": self.n_estimators, "random_state": self.random_state}


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self