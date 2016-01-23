from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
X,y = make_blobs(n_samples=50, centers=2, random_state=0,cluster_std=0.60)

def drawBounderies():
		xfit = np.linspace(-1, 3.5);
		plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
		for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
			yfit = m * xfit + b
			plt.plot(xfit, yfit, '-k')
			plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

		plt.xlim(-1, 3.5);
		plt.show();


def plot_svc_decision_function(clf, ax=None):
	if ax is None:
		ax = plt.gca()
	x = np.linspace(plt.xlim()[0], plt.xlim()[1], 3)
	y = np.linspace(plt.ylim()[0], plt.ylim()[1], 3)
	Y, X = np.meshgrid(y, x)
	P = np.zeros_like(X)
	_Log = open("dec.log",'w')
	for i, xi in enumerate(x):
		   for j, yj in enumerate(y):
			    P[i, j] = clf.decision_function([xi, yj])
			    _Log.write(str(i)+" "+str(j)+" ("+str(xi)+","+str(yj)+") -->"+str(P[i,j])+"\n");
	ax.contour(X, Y, P, colors='k', levels=[-1.0,0.0,1.0], alpha=0.5,linestyles=['--', '-','--']);
	_Log.flush();
	_Log.close();


clf = SVC(kernel='linear');
clf.fit(X,y);
#drawBounderies();
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring');
plot_svc_decision_function(clf);
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
     s=200, facecolors='none');
plt.show();
