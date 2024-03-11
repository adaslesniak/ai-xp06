from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ARandomForest import ARandomForest
from NaiveRandomForest import NaiveRandomForest
import pandas as pd


def evaluate(model, title, predicates, labels):
    train_predicates, test_predicates, train_labels, test_labels = train_test_split(predicates, labels, test_size=0.3, random_state=42)
    model.fit(train_predicates, train_labels)
    predictions = model.predict(test_predicates)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, predictions, average='macro', zero_division=0)
    cv_scores = cross_val_score(model, predicates, labels, cv=5, scoring='accuracy')
    cross_validation_score = f'{cv_scores.mean():.2f} +/- {cv_scores.std():.2f}'
    result = f"avg accuracy: {cross_validation_score},  Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}"
    print(title, ">> ", result)  


def compare(data_name, predicates, labels):
    nest = 25
    rnd = 66
    random_forest_model = RandomForestClassifier(n_estimators=nest, random_state=rnd)
    random_forest_uniform = ARandomForest(n_estimators=nest, random_state=rnd)
    naive_forest = NaiveRandomForest(n_estimators=nest, random_state=rnd)
    print("==========", data_name, "==========")
    evaluate(random_forest_model, "classic", predicates, labels)
    evaluate(random_forest_uniform, "uniform", predicates, labels)
    evaluate(naive_forest, "naive  ", predicates, labels)


#iris = load_iris()
#compare("IRIS", iris.data, iris.target)

wine = pd.read_csv('wine/winequality-red.csv', delimiter=';')
print(wine.info())
print(wine.shape)
wine_features = wine.drop('quality', axis=1).values
wine_labels = wine['quality'].values
compare("Wine Quality", wine_features, wine_labels)



