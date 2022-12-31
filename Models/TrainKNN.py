import numpy as np
from os import listdir
from os.path import isfile, join
import math
from sklearn import preprocessing, model_selection, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report, confusion_matrix

def takeInput(classes):
    TrainX, TrainY = np.zeros((1, 28, 28)), np.zeros((1,), dtype='int8')
    for i, j in enumerate(classes):
        x = np.load('dataset/' + j + '.npy')
        x = x[:10000]
        x = x/255
        x = np.reshape(x, (x.shape[0], 28, 28))
        TrainX = np.append(TrainX, x, axis=0)
        y = np.ones((x.shape[0]), dtype='int8') * i
        TrainY = np.append(TrainY, y)
    TrainX, TrainY = TrainX[1:], TrainY[1:]
    return TrainX, TrainY


def loadClasses(path):
    classes = [f.split(".")[0] for f in listdir(path) if isfile(join("dataset/", f))]
    return classes


def main():
	datasetDir = "dataset/"
	classes = loadClasses(datasetDir)
	clf = KNeighborsClassifier(n_jobs=-1)
	TotalTestX, TotalTestY = np.zeros((1, 784)), np.zeros((1, ), dtype='int8')
	TrainX, TrainY = takeInput(classes)
	TrainX = np.reshape(TrainX, (TrainX.shape[0], 784))

	TrainX, TrainY = shuffle(TrainX, TrainY)

	train_x, test_x, train_y, test_y = train_test_split(TrainX, TrainY, random_state=0, test_size=0.1)
	TotalTestX = np.append(TotalTestX, test_x, axis=0)
	TotalTestY = np.append(TotalTestY, test_y)

	clf.fit(train_x, train_y)

	accuracy = clf.score(test_x, test_y)
	print(accuracy)

	filename = 'model/KNNModel.sav'
	pickle.dump(clf, open(filename, 'wb'))


main()