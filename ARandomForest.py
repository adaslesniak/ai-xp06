import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from UniformSubsampling import UniformSubsampler

class ARandomForest:
    def __init__(self, n_estimators=100, random_state=42, max_features='sqrt'):
        self.n_estimators = n_estimators
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
            tree.fit(sample_X[:, self.feature_indices[i]], sample_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X[:, features_idx]) for tree, features_idx in zip(self.trees, self.feature_indices)])
        final_predictions = stats.mode(predictions, axis=0).mode
        return np.squeeze(final_predictions)