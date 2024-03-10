from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ARandomForest import ARandomForest


def evaluate(model, title, predicates, labels):
    train_predicates, test_predicates, train_labels, test_labels = train_test_split(predicates, labels, test_size=0.3, random_state=42)
    model.fit(train_predicates, train_labels)
    predictions = model.predict(test_predicates)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, predictions, average='macro')
    result = f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}"
    print(title, ">> ", result)  


def compare(data_name, predicates, labels):
    nest = 50
    rnd = 44
    random_forest_model = RandomForestClassifier(n_estimators=nest, random_state=rnd)
    random_forest_uniform = ARandomForest(n_estimators=nest, random_state=rnd)
    print("==========", data_name, "==========")
    evaluate(random_forest_model, "classic", predicates, labels)
    evaluate(random_forest_uniform, "uniform", predicates, labels)


iris = load_iris()
compare("IRIS", iris.data, iris.target)



