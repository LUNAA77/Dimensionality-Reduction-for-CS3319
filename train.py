import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

print("Loading data...")
X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')
print("Data loaded.\n")

# use K-fold cross-validation within the training set to determine hyper-parameters, such as C in SVM.
print("Finding the best C through K-fold cross-validation...")
C_range = [1e-4, 1e-3, 1e-2, 1e-1, 1]
best_C = None
best_score = 0
for C in C_range:
    print('C:', C)
    clf = SVC(C=C, kernel='linear', random_state=42)
    # 5-fold cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    score = np.mean(scores)
    print('Score:', score, '\n')
    if score > best_score:
        best_score = score
        best_C = C
print('Best C:', best_C)

# evaluate the model with the best C
model = SVC(kernel='linear', random_state=42, C=best_C)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
