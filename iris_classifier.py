from sklearn.metrics import confusion_matrix, accuracy_score
...
# Add to evaluation section
y_pred = model.predict(X_test)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))