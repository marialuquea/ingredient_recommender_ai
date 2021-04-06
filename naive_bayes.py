import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix
from data.split_train_test import *

# Load Data
X_train, X_val, X_test, y_train, y_val, y_test = split_data('hold-out')
class_labels = get_labels()

print(class_labels)

print("Training features shape: ")
print(X_train.shape)
print("Validation features shape: ")
print(X_val.shape)
print("Testing features shape: ")
print(X_test.shape)

# Train classifier
clf = CategoricalNB(min_categories = 709)
clf.fit(X_train, y_train)

train_preds = clf.predict(X_train)
val_preds = clf.predict(X_val)

print("Accuracy in train set: ")
print(np.mean(train_preds == y_train))

print("Accuracy in validation set: ")
print(np.mean(val_preds == y_val))

print("Confusion matrix")
print(confusion_matrix(y_val, val_preds, labels = np.unique(y_val)))