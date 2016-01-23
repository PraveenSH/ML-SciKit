from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()

n_samples, n_features = iris.data.shape

#print(iris.keys())
#print((n_samples,n_features)) 
#print(iris.data.shape) #data dimension 2d
#print(iris.target.shape) #class dimension 1d
#print(iris.target_names) #class names
#print(iris.feature_names) #feature names
#print(iris.data) #actual data

x_index = 1
y_index = 3

formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.scatter(iris.data[:,x_index],iris.data[:,y_index],c=iris.target,cmap=plt.cm.get_cmap('RdYlBu',3))
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.clim(-0.5,2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.show()
