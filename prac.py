
from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1., 1.5, -1.],
                    [ 2.,  0., 0.5, 0.5],
                    [ 0.,  1., 2., 2.]])
X_train = np.transpose(X_train)
#scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X_train)

#[-1,1]
scaler = preprocessing.MaxAbsScaler()
X = scaler.fit_transform(X_train)
X_new = np.transpose(X)
#_std=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
#_scaled=X_std/(max_val-min_val)+min_val
print(X_train.shape)
print(X_train)
print(X)
print(X_new)
