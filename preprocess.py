import numpy as np
from sklearn.model_selection import train_test_split

features_path = 'data/AwA2-features.txt'
labels_path = 'data/AwA2-labels.txt'

# Load the features and labels
features = np.loadtxt(features_path)
labels = np.loadtxt(labels_path)
print("Features:", features.shape)
print("Labels:", labels.shape)

# Split the images in each category into 60% for training and 40% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.4, random_state=42)

# save the training and testing data
np.save('processed_data/X_train.npy', X_train)
np.save('processed_data/X_test.npy', X_test)
np.save('processed_data/y_train.npy', y_train)
np.save('processed_data/y_test.npy', y_test)
