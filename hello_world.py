from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data():
    iris = load_iris()
    X, y = train_test_split(iris.data, iris.target, test_size=0.2)
    return X, y

def train_model(X, y, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'svm':
        model = SVC()
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    return {
        'accuracy': accuracy_score(y, predictions),
        'confusion_matrix': confusion_matrix(y, predictions).tolist()
    }

def main():
    X, y = load_data()
    models = ['random_forest', 'svm', 'knn']
    for model_type in models:
        model = train_model(X, y, model_type)
        results = evaluate_model(model, X, y)
        print(f"{model_type.upper()} - Accuracy: {results['accuracy']:.2f}")
        print(f"Confusion Matrix:\n{results['confusion_matrix']}\n")

if __name__ == "__main__":
    main()