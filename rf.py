import numpy as np
import matplotlib.pyplot as plt
import seaborn; 
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,random_state=0,cluster_std=1.0);
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='rainbow');
plt.show();
