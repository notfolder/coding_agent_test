import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_iris_classifier():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nModel Accuracy: {accuracy:.2f}')
    print('Classification report:\n', classification_report(y_test, y_pred))
    
    return model, accuracy

# Example usage
if __name__ == '__main__':
    model, accuracy = train_iris_classifier()
    print(f'\nTrained model with accuracy: {accuracy:.2f}')