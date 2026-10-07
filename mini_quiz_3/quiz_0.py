from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('quiz_file_0.txt')
variances = []

for n in range(1,10):
    km = KMeans(n)

    km.fit(X)

    variances.append(km.inertia_)
    print(n)
    
plt.figure()
plt.plot(range(1,10), variances, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Within-cluster sum of squares (inertia)')
plt.title('Elbow method for K-means')
plt.grid(True)
plt.show()

