import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from UniformSubsampling import UniformSubsampler
import random

class NaiveRandomForest:
    def __init__(self, n_estimators=100, random_state=42, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.random_state = random_state
        random.seed = self.random_state
        np.random.seed = self.random_state
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []  # Store feature indices for each tree


    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            n_features_to_use = int(np.sqrt(n_features))
        else:
            n_features_to_use = self.max_features  # or any other criteria

        feature_sampler = UniformSubsampler(X)
        self.feature_indices = feature_sampler.subsets_with_uniform_distribution(self.n_estimators, n_features_to_use)

        for i in range(self.n_estimators):
            # Bootstrap sample
            sample_X, sample_y = resample(X, y)
            # Train Decision Tree on sampled data with selected features
            tree = DecisionTreeClassifier()
            selected_features = np.random.choice(range(n_features), n_features_to_use, replace=False)
            tree.fit(sample_X[:, selected_features], sample_y)
            self.trees.append(tree)


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