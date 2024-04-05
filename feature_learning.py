import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import DictionaryLearning

print("Loading data...")
X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')
print("Data loaded.\n")

best_C = 0.001

# origin dimension: 2048
embedding_size_range = [8, 16, 32, 64, 128, 256, 512, 1024]

# 假设我们想将数据降维到64维
for n_components in embedding_size_range:
    print(f"Reducing dimensionality to {n_components} using Sparse Coding...")
    dict_learner = DictionaryLearning(
        n_components=n_components, random_state=42, transform_algorithm='lasso_lars')
    X_train_sparse = dict_learner.fit_transform(X_train)
    X_test_sparse = dict_learner.transform(X_test)

    # 评估特征学习后的模型性能
    model = SVC(kernel='linear', random_state=42, C=best_C)
    model.fit(X_train_sparse, y_train)
    accuracy = model.score(X_test_sparse, y_test)
    print("Accuracy after feature learning with Sparse Coding:", accuracy)

# # 使用t-SNE进行特征学习
# embedding_size = 2
# print("Reduce the dimensionality to {} using t-SNE...".format(embedding_size))
# tsne = TSNE(n_components=embedding_size, random_state=42)
# X_tsne = tsne.fit_transform(np.vstack((X_train, X_test)))
# X_train_tsne = X_tsne[:len(X_train)]
# X_test_tsne = X_tsne[len(X_train):]

# # 评估特征学习后的模型性能
# model = SVC(kernel='linear', random_state=42, C=best_C)
# model.fit(X_train_tsne, y_train)
# accuracy = model.score(X_test_tsne, y_test)
# print("Accuracy after feature learning:", accuracy)
