import numpy as np
from os import listdir
from os.path import isfile, join
import math
from sklearn import preprocessing, model_selection, neighbors
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

def takeInput(classes, k, test=0):
	TrainX, TrainY = np.zeros((1, 28, 28)), np.zeros((1,), dtype='int8')
	for i, j in enumerate(classes):
		x = np.load('dataset/' + j + '.npy')
		x = x/255
		x = x[10000*(k):10000*(k+1)]
		x = np.reshape(x, (x.shape[0], 28, 28))
		TrainX = np.append(TrainX, x, axis=0)
		if test==0:
			y = np.load('KMeanY/' + classes[i] + '.npy')
			y = y[10000*(k):10000*(k+1)]
		elif test==1:
			y = np.ones((x.shape[0]), dtype='int8') * i
		print(str(x.shape) + ' ' + str(y.shape) + ' ' + str(k))
		TrainY = np.append(TrainY, y)
	TrainX, TrainY = TrainX[1:], TrainY[1:]
	return TrainX, TrainY


def loadClasses(path):
	classes = [f.split(".")[0] for f in listdir(path) if isfile(join("dataset/", f))]
	return classes


def main():
	datasetDir = "dataset/"
	classes = loadClasses(datasetDir)
	clf = SGDClassifier(verbose=True, max_iter=1000)
	TotalTestX, TotalTestY = np.zeros((1, 784)), np.zeros((1, ), dtype='int8')

	for i in (range(12)):
		TrainX, TrainY = takeInput(classes, i, 1)
		TrainX = np.reshape(TrainX, (TrainX.shape[0], 784))

		TrainX, TrainY = shuffle(TrainX, TrainY)
		print(str(TrainX.shape) + ' ' + str(TrainY.shape))

		train_x, test_x, train_y, test_y = train_test_split(TrainX, TrainY, random_state=0, test_size=0.1)
		TotalTestX = np.append(TotalTestX, test_x, axis=0)
		TotalTestY = np.append(TotalTestY, test_y)

		clf.partial_fit(train_x, train_y, classes=np.unique(train_y))
		accuracy = clf.score(test_x, test_y)
		print(str(i+1) + ' ' + str(accuracy))

	TotalTestX, TotalTestY = TotalTestX[1:], TotalTestY[1:]
	TotalTestX = np.reshape(TotalTestX, (TotalTestX.shape[0], 784))
	
	print("Test Accuracy: " + str(clf.score(TotalTestX, TotalTestY)))

	accuracy = clf.score(TotalTestX, TotalTestY)

	print("TOTAL ACCURACY IS: " + str(accuracy) + '_____________________')

	filename = 'model/SVMmodel.sav'
	pickle.dump(clf, open(filename, 'wb'))


main()