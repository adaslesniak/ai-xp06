import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# A method to check how feature is distributed against feature importance
# Results cleary show that sci-kit algorithm for random forest has focus mechanism
# It represents more prominent features more often. Now there is still some other bias or focus:
# even among equally important features some are represented much more often than others.
def analyse_usage(predicates, labels, data_title):
    random_states = [0, 42, 44, 99, 12345, 2024, 1981]
    importances = np.zeros(predicates.shape[1])
    for meaning_of_world in random_states:
        sk_rf = RandomForestClassifier(n_estimators=100, random_state=meaning_of_world)
        sk_rf.fit(predicates, labels)
        # Analyze features used across all trees
        feature_usage = [tree.tree_.feature for tree in sk_rf.estimators_]
        unique, counts = np.unique(np.concatenate(feature_usage), return_counts=True)
        filtered_counts = [count for feature, count in zip(unique, counts) if feature != -2]
        print("Feature Usage Distribution for " + data_title + ":", filtered_counts)
        importances += sk_rf.feature_importances_
    average_importances = importances / len(random_states)
    rounded_importances = [round(importance * 100, 1) for importance in average_importances]
    print("Feature Importance for         " + data_title + ":", rounded_importances)

iris = load_iris()
iris_predicates = iris.data
iris_labels = iris.target
analyse_usage(iris_predicates, iris_labels, "IRIS")


wine = pd.read_csv('wine/winequality-red.csv', delimiter=';')
wine_features = wine.drop('quality', axis=1).values
wine_labels = wine['quality'].values
analyse_usage(wine_features, wine_labels, "WINE QUALITY")
