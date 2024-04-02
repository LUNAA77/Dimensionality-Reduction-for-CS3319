import numpy as np

X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# use K-fold cross-validation within the training set to determine hyper-parameters, such as C in SVM.
# K = 5
# C_values = [0.01, 0.1, 1, 10, 100]
# best_C = None
# best_score = 0
# for C in C_values:
#     clf = SVC(C=C, kernel='linear', random_state=42)
#     scores = cross_val_score(clf, X_train, y_train, cv=K)
#     score = np.mean(scores)
#     print('C:', C, 'Score:', score)
#     if score > best_score:
#         best_score = score
#         best_C = C
# print('Best C:', best_C)
# best_C = 0.01
