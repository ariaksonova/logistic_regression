import pandas
import argparse

def normalize(X):
	for i in range(len(X)):
		X[i] = ( X[i] - X.mean())  / X.std()
	return X

def find_all_num(dataset):
	data_name = []
	for x in dataset.columns:
		try:
			float(dataset[x][0])
			data_name.append(x)
		except ValueError:
			continue
	data_name.append("Hogwarts House")
	if "Defense Against the Dark Arts" in data_name:
		data_name.remove("Defense Against the Dark Arts")
	if "Arithmancy" in data_name:
		data_name.remove("Arithmancy")
	if "Care of Magical Creatures" in data_name:
		data_name.remove("Care of Magical Creatures")
	return(data_name)

def read_data(argv, my_str):
	parse = argparse.ArgumentParser(usage=my_str)
	parse.add_argument('path_to_csv_file', type=str, help="It is path to your file for train model.")
	filename = parse.parse_args(argv).path_to_csv_file
	dataset = pandas.read_csv(filename, sep=",", index_col="Index")
	return(dataset)

def read_data_for_train(args, my_str):
    parser = argparse.ArgumentParser(usage=my_str)
    parser.add_argument('path_to_csv_file',
                        type=str,
                        help="It is path to your file for train model.")
    parser.add_argument('-o',
                        type=str,
                        help="It is output file with weights.",
                        default="result")
    parser.add_argument('-n',
                        type=int,
                        help='It is number iterations.',
                        default=25)
    parser.add_argument('-lr',
                        type=float,
                        help='It is learning rate.',
                        default=0.2)
    parser.add_argument('-acc',
                        type=float,
                        help='It is accuracy.',
                        default=0.1)
    args = parser.parse_args(args)
    dataset = pandas.read_csv(args.path_to_csv_file, index_col = "Index")
    return dataset, args.o, args.n, args.lr, args.acc


def read_data_for_test(args, my_str):
	parser = argparse.ArgumentParser(usage=my_str)
	parser.add_argument('path_to_csv_file',
						type=str,
						help='It is path to your file for test model."')
	parser.add_argument('-s',
						type=str,
						help='It is output file name.',
						default='houses.csv')
	parser.add_argument('-w',
						type=str,
						help='It is file with weights.',
						default="result.npy")
	args = parser.parse_args(args)
	dataset = pandas.read_csv(args.path_to_csv_file, index_col = "Index")
	return dataset, args.s, args.w