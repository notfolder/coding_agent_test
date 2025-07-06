from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

def classify_iris():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = KNeighborsClassifier()
    model.fit(X, y)
    sample = [[5.0, 3.0, 2.0, 1.0]]
    prediction = model.predict(sample)
    return f'Iris species: {iris.target_names[prediction[0]]}'

if __name__ == "__main__":
    print(classify_iris())