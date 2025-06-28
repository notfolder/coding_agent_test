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
    print("Confusion Matrix:\n", confusion_matrix(y, predictions))
    print("Accuracy:", accuracy_score(y, predictions))
    return model.predict(iris.data[:1])

if __name__ == "__main__":
    print("Prediction:", classify_iris())