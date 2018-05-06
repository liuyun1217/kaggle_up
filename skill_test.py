import sklearn.preprocessing as preprocessing
import numpy as np

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
x1 = np.array([1,2,3,4,5,6,7,8,9])
#scaler = preprocessing.StandardScaler().fit(x1)
scaler1 = preprocessing.scale(x1)
#print(x1)
#print(scaler.set_params())
print(scaler1)