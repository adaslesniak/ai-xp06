from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ARandomForest import ARandomForest
from RandomSubspaces import RandomSubspacesMethod
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


def evaluate(model, title, predicates, labels):
    train_predicates, test_predicates, train_labels, test_labels = train_test_split(predicates, labels, test_size=0.3, random_state=42)
    model.fit(train_predicates, train_labels)
    predictions = model.predict(test_predicates)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, predictions, average='macro', zero_division=0)
    cv_scores = cross_val_score(model, predicates, labels, cv=5, scoring='accuracy')
    cross_validation_score = f'{cv_scores.mean():.2f} +/- {cv_scores.std():.2f}'
    result = f"cv accuracy: {cross_validation_score},  Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}"
    print(title, ">> ", result)  


def compare(data_name, predicates, labels):
    nest = 25
    rnd = 66
    depth = 5
    n_features = predicates.shape[1]
    random_forest_model = RandomForestClassifier(n_estimators=nest, random_state=rnd, max_depth=depth)
    random_forest_uniform = ARandomForest(n_estimators=nest, random_state=rnd, max_depth=depth)
    subspaces = RandomSubspacesMethod(n_estimators=nest, random_state=rnd, max_features='sqrt')
    bagging = BaggingClassifier(estimator= DecisionTreeClassifier(), n_estimators=nest, random_state=rnd, bootstrap=False, max_features=int(np.sqrt(n_features)))
    print("==========" + data_name + str(n_features) + "features]==========")
    evaluate(random_forest_model,   "random forest     ", predicates, labels)
    evaluate(random_forest_uniform, "adjusted subspaces", predicates, labels)
    evaluate(subspaces, "random subspaces  ", predicates, labels)
    evaluate(bagging, "bagging           ", predicates, labels)

iris = load_iris()
compare("IRIS", iris.data, iris.target)

wine = pd.read_csv('wine/winequality-red.csv', delimiter=';')
wine_features = wine.drop('quality', axis=1).values
wine_labels = wine['quality'].values
compare("Wine Quality", wine_features, wine_labels)

human_activity = pd.read_csv("har_data.csv")
har_features = human_activity.drop(['activity', 'subject'], axis=1).values
har_labels = human_activity['activity'].values
compare("Human Activity", har_features, har_labels)



