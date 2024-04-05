import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.model_selection import cross_val_score

print("Loading data...")
X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')
print("Data loaded.\n")

best_C = 0.001
# origin dimension: 2048

# # 使用SelectKBest进行特征选择
# n_component_range = [8, 16, 32, 64, 128, 256, 512, 1024]
# for n_components in n_component_range:
#     print("Reducing dimensionality to {} using SelectKBest...".format(n_components))
#     selector = SelectKBest(chi2, k=n_components)
#     X_train_selected = selector.fit_transform(X_train, y_train)
#     X_test_selected = selector.transform(X_test)
#     print("X_train_selected shape:", X_train_selected.shape)
#     print("X_test_selected shape:", X_test_selected.shape)

#     # 评估特征选择后的模型性能
#     model = SVC(kernel='linear', random_state=42, C=best_C)
#     model.fit(X_train_selected, y_train)
#     accuracy = model.score(X_test_selected, y_test)
#     print("Accuracy after feature selection with SelectKBest:", accuracy)

# 使用 RFE 进行特征选择
n_components = 8
print("Reduce the dimensionality to {} using RFE...".format(n_components))
selector = RFE(SVC(kernel='linear', random_state=42, C=best_C),
               n_features_to_select=n_components, step=128)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
print("X_train_selected shape:", X_train_selected.shape)
print("X_test_selected shape:", X_test_selected.shape)

# 评估特征选择后的模型性能
model = SVC(kernel='linear', random_state=42, C=best_C)
model.fit(X_train_selected, y_train)
accuracy = model.score(X_test_selected, y_test)
print("Accuracy after feature selection:", accuracy)
