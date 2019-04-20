import sys
import matplotlib.pyplot as plt
import pandas

from parse_data import *

def histogram(argv):
	try:
		dataset = read_data(argv, "python3 histogram.py [-h] path_to_csv_file")
		tmp = pandas.to_datetime(dataset['Birthday'])
		dataset = dataset.assign(Course=[x.year for x in tmp])
		all_course = dataset.Course.unique()
		all_course = sorted(all_course)
		plt.hist([dataset.loc[dataset['Hogwarts House'] == 'Gryffindor']['Course'],dataset.loc[dataset['Hogwarts House'] == 'Hufflepuff']['Course'],
			dataset.loc[dataset['Hogwarts House'] == 'Slytherin']['Course'],dataset.loc[dataset['Hogwarts House'] == 'Ravenclaw']['Course']],
			bins=all_course, color=['r','y','g','b'])
		print(dataset.loc[dataset['Hogwarts House'] == 'Gryffindor']['Course'])
		plt.show()
	except Exception:
		print("Hmmmmmm... Something went wrong...")

if __name__ == "__main__":
	histogram(sys.argv[1:])