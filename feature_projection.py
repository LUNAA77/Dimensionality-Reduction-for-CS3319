import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

print("Loading data...")
X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')
print("Data loaded.\n")

best_C = 0.001

# origin dimension: 2048
n_components_range = [8, 16, 32, 64, 128, 256, 512, 1024]

# # 使用linear PCA进行特征投影
# for n_components in n_components_range:
#     print("Reduce the dimensionality to {} using Linear PCA...".format(n_components))
#     pca = PCA(n_components=n_components, random_state=42)
#     X_train_pca = pca.fit_transform(X_train)
#     X_test_pca = pca.transform(X_test)

#     # 评估特征投影后的模型性能
#     model = SVC(kernel='linear', random_state=42, C=best_C)
#     model.fit(X_train_pca, y_train)
#     accuracy = model.score(X_test_pca, y_test)
#     print("Accuracy after feature projection:", accuracy)

# 使用LDA进行特征投影
for n_components in n_components_range:
    print("Reduce the dimensionality to {} using LDA...".format(n_components))
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # 评估特征投影后的模型性能
    model = SVC(kernel='linear', random_state=42, C=best_C)
    model.fit(X_train_lda, y_train)
    accuracy = model.score(X_test_lda, y_test)
    print("Accuracy after feature projection:", accuracy)
