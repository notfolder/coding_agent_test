from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def classify_iris():
    iris = load_iris()
    X, y = train_test_split(iris.data, iris.target, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f'Accuracy: {accuracy_score(y, y_pred):.2f}')
    print('Confusion Matrix:\n', confusion_matrix(y, y_pred))
    return y_pred

def main():
    print("hello world")
    print("Iris classification result:", classify_iris())

if __name__ == "__main__":
    main()