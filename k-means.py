import numpy as np
import matplotlib.pyplot as plt
import seaborn; 
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl
from sklearn import neighbors, datasets
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs;
X,y = make_blobs(n_samples=300,centers=4,random_state=0,cluster_std=0.60);
plt.scatter(X[:, 0], X[:, 1],s=50);

est = KMeans(4);
est.fit(X);
#X_pred = [[1.0,1.0],[2.0,2.0],[10,10],[20,20],[-1,-1]]
X_pred =np.array([[-2,3],[1,5],[1.0,1.0],[2.0,2.0],[10,1],[-1,8]])
print X_pred

y_kmeans = est.predict(X_pred);
plt.scatter(X_pred[:, 0], X_pred[:, 1], c=y_kmeans, s=50, cmap='rainbow');
plt.show();
