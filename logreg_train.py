import sys
import numpy
import copy
import math

from parse_data import *

def find_hipotesis(vars, weights, data_name):
    res = weights['bias']
    for key in data_name:
        res += vars[key] * weights[key]
    return 1 / (1 + math.exp((-res)))

def check_acc(old, new, acc):
    for key in old.keys():
        if abs(old[key] - new[key]) >= accuracy:
            return 1
    return 0

def update_weights(weights, dataset, l_rate, house_num, data_name):
    for i, row in dataset.iterrows():
        res = 0 if row["Hogwarts House"] != house_num else 1
        hipotesis = find_hipotesis(row.to_dict(), weights, data_name)
        for col_name in data_name:
            weights[col_name] += (l_rate * (res - hipotesis) * hipotesis * (1 - hipotesis) * row[col_name])
        weights['bias'] += l_rate * (res - hipotesis)
    return weights

def house_learn(dataset, l_rate, x, n_iter, accurancy, data_name):
	weights = dict(map(lambda x: (x, 0), data_name + ['bias', ]))
	for i in range(n_iter):
		new_weights = update_weights(copy.deepcopy(weights), dataset, l_rate, x, data_name)
		if check_acc(weights, new_weights, accurancy) == 1:
			break
		weights = new_weights
	return weights

def get_weights(dataset, output_file, n_iter, l_rate, accurancy, data_name):
	weights = []
	for x in range(4):
		weights.append(house_learn(dataset, l_rate, x, n_iter, accurancy, data_name))
	return weights

def logreg_train(argv):
	dataset, output_file, n_iter, l_rate, accurancy = read_data_for_train(argv, 'python3 logreg_train.py [-h] [-o O] [-n N] [-lr LR] [-acc ACC] path_to_csv_file')
	dataset, data_name = normalize(dataset)
	weights = get_weights(dataset, output_file, n_iter, l_rate, accurancy, data_name)

if __name__ == "__main__":
	logreg_train(sys.argv[1:])