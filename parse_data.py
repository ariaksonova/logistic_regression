import pandas
import argparse

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
                        help="It is output json file with weights.",
                        default="result.json")
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
    dataset = pandas.read_csv(args.path_to_csv_file, usecols=[
    'Astronomy',
    'Herbology',
    'Divination',
    'Muggle Studies',
    'History of Magic',
    'Transfiguration',
    'Charms',
    'Flying',
    'Hogwarts House', ])
    return dataset, args.o, args.n, args.lr, args.acc

def normalize(dataset):
	data_name = find_all_num(dataset)
	data_name.remove("Hogwarts House")
	tmp_dataset = dataset[data_name]
	tmp_dataset = tmp_dataset.dropna()
	result = tmp_dataset.copy()
	for x in data_name:
		max_value = tmp_dataset[x].max()
		min_value = tmp_dataset[x].min()
		result[x] = ((tmp_dataset[x] - min_value) / (max_value - min_value))
	return result, data_name