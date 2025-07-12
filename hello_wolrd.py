from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

def train_classifier():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model, iris.target_names

def main():
    model, class_names = train_classifier()
    print(f"Trained classifier with {len(model.classes_)} classes:")
    print(class_names)

if __name__ == "__main__":
    main()