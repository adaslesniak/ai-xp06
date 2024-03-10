import numpy as np
import pandas as pd
import random

class UniformSubsampler:

    def __init__(self, data:pd.DataFrame):    
        self._original_features = data.columns.tolist()
        self._inflated_features = data.columns.tolist()
        self._org_nr_of_features = data.shape[1]
        

    def _sample_with_uniform_distribution(self, how_many_to_select):
        features_subset = []
        while len(features_subset) < how_many_to_select:
            feature = random.choice(self._inflated_features)
            if feature in features_subset:
                continue
            features_subset.append(feature)
            if self._inflated_features.count(feature) == 1: 
                self._inflated_features += self._original_features #ensure feature is not completly removed
            self._inflated_features.remove(feature) # reduce relative chances of selected feature to be reselected again
        return features_subset

    def subsets_with_uniform_distribution(self, nr_of_subsets=100, features_per_subset = None):
        if features_per_subset is None:
            features_per_subset = int(np.ceil(np.sqrt(self._org_nr_of_features)))
        subsets = []
        for _ in range(nr_of_subsets):
            next_subset = self._sample_with_uniform_distribution(features_per_subset)
            subsets.append(next_subset)
        return subsets



