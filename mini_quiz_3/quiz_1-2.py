import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data1 = np.loadtxt('quiz_file_1.txt')
data2 = np.loadtxt('quiz_file_2.txt')

# X must be 2D: (n_samples, 1)
X = data1.reshape(-1, 1)
y = data2             # y can be 1D

reg = LinearRegression().fit(X, y)

a = reg.coef_[0]      # slope
b = reg.intercept_    # intercept

print(a, b)