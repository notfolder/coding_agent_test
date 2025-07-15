import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def classify_iris():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression()
    model.fit(X, y)
    return model.predict([[5.0, 3.0, 1.0, 0.5]])

def main():
    result = classify_iris()
    print("Predicted species:", result)

if __name__ == "__main__":
    main()