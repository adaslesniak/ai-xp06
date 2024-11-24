import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

def _generate_data(predicates, labels, data_title):
    print("---------" + data_title + "--------")
    random_states = [0, 42, 44, 99, 12345, 2024, 1981]
    importances = np.zeros(predicates.shape[1])
    usages = np.zeros((len(random_states), predicates.shape[1]))
    for idx, meaning_of_world in enumerate(random_states):
        sk_rf = RandomForestClassifier(n_estimators=250, random_state=meaning_of_world)
        sk_rf.fit(predicates, labels)
        # Analyze features used across all trees
        feature_usage = [tree.tree_.feature for tree in sk_rf.estimators_]
        unique, counts = np.unique(np.concatenate(feature_usage), return_counts=True)
        full_counts = np.zeros(predicates.shape[1])
        for feature, count in zip(unique, counts):
            if feature != -2:
                full_counts[feature] = count
        usages[idx] = full_counts
        importances += sk_rf.feature_importances_
    mean_usages = np.mean(usages, axis=0)  # Raw means
    mean_usages[mean_usages == 0] = 1  # Prevent division by zero
    deviations = np.abs(usages - mean_usages)  # Raw deviations
    average_importances = importances / len(random_states)
    rounded_importances = np.array([round(importance * 100, 1) for importance in average_importances])
    percent_deviations = (deviations / mean_usages) * 100
    return usages, rounded_importances, percent_deviations, data_title

def generate_data(predicates, labels, data_title):
    print("---------" + data_title + "--------")
    random_states = [0, 42, 44, 99, 12345, 2024, 1981]
    importances = np.zeros(predicates.shape[1])
    usages = np.zeros((len(random_states), predicates.shape[1]))
    for idx, meaning_of_world in enumerate(random_states):
        sk_rf = RandomForestClassifier(n_estimators=250, random_state=meaning_of_world)
        sk_rf.fit(predicates, labels)
        # Analyze features used across all trees
        feature_usage = [tree.tree_.feature for tree in sk_rf.estimators_]
        unique, counts = np.unique(np.concatenate(feature_usage), return_counts=True)
        full_counts = np.zeros(predicates.shape[1])
        for feature, count in zip(unique, counts):
            if feature != -2:
                full_counts[feature] = count
        usages[idx] = full_counts
        importances += sk_rf.feature_importances_
    mean_usages = np.mean(usages, axis=0)  # Raw means
    mean_usages[mean_usages == 0] = 1  # Prevent division by zero
    deviations = np.abs(usages - mean_usages)  # Raw deviations
    average_importances = importances / len(random_states)
    rounded_importances = np.array([round(importance * 100, 1) for importance in average_importances])
    percent_deviations = (deviations / mean_usages) * 100
    rounded_deviations = np.round(percent_deviations, 1)
    return usages, rounded_importances, rounded_deviations

def iris():
    iris = load_iris()
    return generate_data(iris.data, iris.target, "IRIS")

# https://archive.ics.uci.edu/dataset/186/wine+quality
def wine_quality():
    wine = pd.read_csv('wine/winequality-red.csv', delimiter=';')
    wine_features = wine.drop('quality', axis=1).values
    wine_labels = wine['quality'].values
    return generate_data(wine_features, wine_labels, "WINE QUALITY")

def breast_cancer():
    bc = load_breast_cancer()
    return generate_data(bc.data, bc.target, "CANCER")

# https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
def human_activity():
    human_activity = pd.read_csv("har_data.csv")
    har_predictors = human_activity.drop(['activity', 'subject'], axis=1).values
    har_labels = human_activity['activity'].values
    return generate_data(har_predictors, har_labels, "HUMAN ACTIVITY")