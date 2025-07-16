from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_train():
    iris = load_iris()
    X, y = train_test_split(iris.data, iris.target, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, X, y

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    return accuracy_score(y, predictions)

def main():
    model, X, y = load_and_train()
    accuracy = evaluate_model(model, X, y)
    print(f"Model accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()