import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

print("Loading data...")
X_train = np.load('processed_data/X_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')
print("Data loaded.\n")

best_C = 0.001

# origin dimension: 2048

# # 使用 RFE 进行特征选择
# n_components = 32
# print("Reduce the dimensionality to {} using RFE...".format(n_components))
# selector = RFE(SVC(kernel='linear', random_state=42, C=best_C),
#                n_features_to_select=n_components, step=128)
# X_train_selected = selector.fit_transform(X_train, y_train)
# X_test_selected = selector.transform(X_test)
# print("X_train_selected shape:", X_train_selected.shape)
# print("X_test_selected shape:", X_test_selected.shape)

# # 评估特征选择后的模型性能
# model = SVC(kernel='linear', random_state=42, C=best_C)
# model.fit(X_train_selected, y_train)
# accuracy = model.score(X_test_selected, y_test)
# print("Accuracy after feature selection:", accuracy)

# 使用遗传算法进行特征选择


def genetic_algorithm(X_train, X_test, y_train, y_test):
    """
    使用遗传算法进行特征选择
    :param X_train: 训练集特征
    :param X_test: 测试集特征
    :param y_train: 训练集标签
    :param y_test: 测试集标签
    :return: 最佳个体
    """
    # Initialize the population
    population_size = 64
    population = []
    for _ in range(population_size):
        mask = np.random.choice([True, False], size=X_train.shape[1])
        population.append(mask)

    # Evaluate the fitness of each individual in the population
    fitness_scores = []
    for mask in population:
        X_train_masked = X_train[:, mask]
        X_test_masked = X_test[:, mask]
        model = RandomForestClassifier(random_state=42)
        scores = cross_val_score(model, X_train_masked, y_train, cv=5)
        fitness_scores.append(scores.mean())

    # Select the best individuals for reproduction
    num_parents = 2
    parents = []
    for _ in range(num_parents):
        max_fitness_idx = np.argmax(fitness_scores)
        parents.append(population[max_fitness_idx])
        fitness_scores[max_fitness_idx] = -1

    # Crossover
    offspring = []
    for _ in range(population_size - num_parents):
        parent1 = parents[0]
        parent2 = parents[1]
        crossover_point = np.random.randint(0, len(parent1))
        child = np.concatenate(
            (parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)

    # Mutation
    mutation_rate = 0.1
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if np.random.rand() < mutation_rate:
                offspring[i][j] = not offspring[i][j]

    # Combine parents and offspring to form the new population
    new_population = parents + offspring

    # Select the best individual from the new population
    best_individual = None
    best_fitness = -1
    for mask in new_population:
        X_train_masked = X_train[:, mask]
        X_test_masked = X_test[:, mask]
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_masked, y_train)
        y_pred = model.predict(X_test_masked)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_fitness:
            best_fitness = accuracy
            best_individual = mask

    return best_individual


best_individual = genetic_algorithm(
    X_train, X_test, y_train, y_test)
X_train_selected = X_train[:, best_individual]
X_test_selected = X_test[:, best_individual]
print("X_train_selected shape after genetic algorithm:", X_train_selected.shape)
print("X_test_selected shape after genetic algorithm:", X_test_selected.shape)
model = SVC(kernel='linear', random_state=42, C=best_C)
model.fit(X_train_selected, y_train)
accuracy = model.score(X_test_selected, y_test)
print("Accuracy after genetic algorithm:", accuracy)
