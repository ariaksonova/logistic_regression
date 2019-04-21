import sys
import numpy
import pandas
from collections import OrderedDict

from parse_data import *

def preprocessing(dataset):
	tmp_dataset = dataset.iloc[:,5:]
	tmp_dataset = tmp_dataset.dropna()
	features = numpy.array(tmp_dataset)
	numpy.apply_along_axis(normalize, 0, features)
	return features

def predict(dataset, file_weights, output_file):
	data = preprocessing(dataset)
	weights = numpy.load(file_weights)
	result = [max((i.dot(w), c) for w, c in weights)[1] for i in numpy.insert(data, 0, 1, axis=1)]
	print('Done!')
	print(f'Result has been saved to "{output_file}"')
	houses = pandas.DataFrame(OrderedDict ({'Index':range(len(result)), 'Hogwarts House':result}))
	houses.to_csv(output_file, index=False)

def logreg_predict(argv):
	try:
		dataset, output_file, file_weights = read_data_for_test(argv, "python3 logreg_predict.py [-h] [-s S] [-w W] path_to_csv_file")
		predict(dataset, file_weights, output_file)
	except Exception:
		print("Hmmmmmm... Something went wrong...")

if __name__ == "__main__":
	logreg_predict(sys.argv[1:])