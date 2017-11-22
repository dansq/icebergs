import pandas as pd
import numpy as np
from sklearn.svm import SVR


train_df = pd.read_json('train.json')
print('passou treino')
test_df = pd.read_json('test.json')
print('passou teste')

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
Y_train = np.array(train_df["is_iceberg"])
vars = X_train.shape
d2_train = X_train.reshape(vars[0], vars[1] * vars[2] * vars[3])

print("Xtrain:", X_train.shape)
print("Ytrain:", Y_train.shape)
print(Y_train)

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
print("Xtest:", X_test.shape)

clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

clf.fit(d2_train, Y_train)

response = clf.predict(X_test)

print('treinou')

print()

print(response)
