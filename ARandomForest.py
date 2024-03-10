import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

class ARandomForest:
    def __init__(self, n_estimators=100, random_state=42, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []  # Store feature indices for each tree

    def fit(self, X, y):
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            # Bootstrap sample
            sample_X, sample_y = resample(X, y)
            
            # Select features based on your uniform distribution algorithm
            if self.max_features == 'sqrt':
                n_features_to_use = int(np.sqrt(n_features))
            else:
                n_features_to_use = self.max_features  # or any other criteria
            features_idx = np.random.choice(range(n_features), n_features_to_use, replace=False)
            self.feature_indices.append(features_idx)
            
            # Train Decision Tree on sampled data with selected features
            tree = DecisionTreeClassifier()
            tree.fit(sample_X[:, features_idx], sample_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X[:, features_idx]) for tree, features_idx in zip(self.trees, self.feature_indices)])
        final_predictions = stats.mode(predictions, axis=0).mode
        return np.squeeze(final_predictions)