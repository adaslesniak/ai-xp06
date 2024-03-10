from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the Iris dataset
iris = load_iris()
predicates = iris.data
labels = iris.target

# Prepare the model
train_predicates, test_predicates, train_labels, test_labels = train_test_split(predicates, labels, test_size=0.3, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest_model.fit(train_predicates, train_labels)

# Evaluate the model
predictions = random_forest_model.predict(test_predicates)
accuracy = accuracy_score(test_labels, predictions)
precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, predictions, average='macro')
result = f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}"
print(result)  
