from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def main():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split dataset into training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Train the model using the training sets
    knn.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    
    # Model Accuracy
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    main()