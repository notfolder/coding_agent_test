from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def classify_iris():
    iris = load_iris()
    X, y = train_test_split(iris.data, iris.target, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X, y)
    return model.predict([iris.data[0]])

def main():
    print("hello world")
    print("Iris classification result:", classify_iris())

if __name__ == "__main__":
    main()