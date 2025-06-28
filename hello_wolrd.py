from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def classify_iris():
    iris = load_iris()
    X, y = train_test_split(iris.data, iris.target, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    predictions = model.predict(X)
    return {
        'confusion_matrix': confusion_matrix(y, predictions).tolist(),
        'accuracy': float(accuracy_score(y, predictions))
    }

if __name__ == '__main__':
    results = classify_iris()
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    print(f"Accuracy: {results['accuracy']:.2f}")