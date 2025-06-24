import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def train_and_classify():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression()
    model.fit(X, y)
    # Example prediction
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)
    print(f"Predicted class: {iris.target_names[prediction[0]]}")

def main():
    train_and_classify()

if __name__ == "__main__":
    main()