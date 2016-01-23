from sklearn import neighbors, datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
print "Clustering..."
knn.fit(X,y)
print "Predicting..."
X_new = [3,6,4,2]
result = knn.predict([X_new,])
print(result)
print(iris.target_names[result])
print(iris.target_names)
print(knn.predict_proba([X_new,]))
