from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predictions):.2f}')

if __name__ == "__main__":
    main()