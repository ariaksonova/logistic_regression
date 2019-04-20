import pandas
import argparse

def read_data(argv, my_str):
	parse = argparse.ArgumentParser(usage=my_str)
	parse.add_argument('path_to_csv_file', type=str, help="It is path to your file for train model.")
	filename = parse.parse_args(argv).path_to_csv_file
	dataset = pandas.read_csv(filename, sep=",")
	return(dataset)