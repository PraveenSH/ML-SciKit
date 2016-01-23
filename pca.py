import numpy as np
import matplotlib.pyplot as plt
import seaborn; 
from sklearn import neighbors, datasets
from sklearn.decomposition import PCA

import pylab as pl

np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T

pca = PCA(n_components=2);
pca.fit(X);
print(pca.explained_variance_);
print(pca.components_);
print(pca.explained_variance_ratio_);

plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.5)
for length, vector in zip(pca.explained_variance_ratio_, pca.components_):
	    v = vector * 3 * np.sqrt(length)
	    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.axis('equal');
plt.show();

