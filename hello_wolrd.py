from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample scikit-learn classifier
dataset = np.array([[1, 2], [2, 3], [3, 4]])
targets = np.array([0, 1, 1])

model = LogisticRegression()
model.fit(dataset, targets)

print("Model trained with scikit-learn")