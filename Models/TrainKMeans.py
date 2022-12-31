from sklearn.cluster import KMeans
from mlxtend.cluster import Kmeans
import numpy as np
from os import listdir
from os.path import isfile, join
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


def loadClasses(path):
    classes = [f.split(".")[0] for f in listdir(path) if isfile(join("dataset/", f))]
    return classes


def K_means(classes):
	for i, j in enumerate(classes):
	    X = np.load('dataset/' + classes[i] + '.npy')

	    # km = KMeans(n_clusters=3, init='k-means++', max_iter=10, verbose=True)
	    km = Kmeans(k=3, print_progress=3)
	    km.fit(X)

	    y_clust = km.predict(X)
	    for k in range(len(y_clust)):
	    	y_clust[k] = y_clust[k] + i*3

	    print()
	    print(np.unique(y_clust))
	    np.save('KMeanY/' + classes[i] + '.npy', y_clust)


def main():
    datadir = 'dataset/'
    x = loadClasses(datadir)
    K_means(x)


main()