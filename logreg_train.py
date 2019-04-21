import sys
import numpy
import copy
import math

from parse_data import *

def preprocessing(dataset):
	tmp = dataset.dropna()
	features = numpy.array((tmp.iloc[:,5:]))
	labels = numpy.array(tmp.loc[:,"Hogwarts House"])
	numpy.apply_along_axis(normalize, 0, features)
	return features, labels

def get_weights(dataset, output_file, n_iter, l_rate, accurancy):
	weights = []
	features, labels = preprocessing(dataset)
	features = numpy.insert(features, 0, 1, axis=1)
	tmp = features.shape[0]
	for i in numpy.unique(labels):
		y_copy = numpy.where(labels == i, 1, 0)
		tmp_weights = numpy.ones(features.shape[1])
		for _ in range(n_iter):
			output = features.dot(tmp_weights)
			errors = y_copy - (1 / (1 + numpy.exp(-output)))
			gradient = numpy.dot(features.T, errors)
			tmp_weights += 5e-5 * gradient
		weights.append((tmp_weights, i))
	numpy.save(output_file, weights)
	return weights

def predict(x, weights):
	return max((x.dot(w), c) for w, c in weights)[1]

def estimate_error(dataset, weights):
	features, labels = preprocessing(dataset)
	return sum([predict(i, weights) for i in numpy.insert(features, 0, 1, axis=1)] == labels) / len(labels)   

def logreg_train(argv):
	try:
		dataset, output_file, n_iter, l_rate, accurancy = read_data_for_train(argv, 'python3 logreg_train.py [-h] [-o O] [-n N] [-lr LR] [-acc ACC] path_to_csv_file')
		weights = get_weights(dataset, output_file, n_iter, l_rate, accurancy)
		print('Model has been trained!')
		print('Accuracy: {:3.2f}%'.format(estimate_error(dataset, weights)))
		print(f'Weights has been saved to "{output_file}"')
	except Exception:
		print("Hmmmmmm... Something went wrong...")

if __name__ == "__main__":
	logreg_train(sys.argv[1:])