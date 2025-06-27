from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def classify_iris():
    iris = load_iris()
    X, y = train_test_split(iris.data, iris.target, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model.predict(iris.data[:1])

if __name__ == "__main__":
    print("Prediction:", classify_iris())