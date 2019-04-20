import sys
import matplotlib.pyplot as plt

from parse_data import *

def scatter_plot(argv):
	try:
		dataset = read_data(argv, "python3 scatter_plot.py [-h] path_to_csv_file")
		plt.scatter(dataset['Defense Against the Dark Arts'], dataset['Astronomy'])
		plt.xlabel('Defense Against the Dark Arts')
		plt.ylabel('Astronomy')
		plt.show()
	except Exception:
		print("Hmmmmmm... Something went wrong...")

if __name__ == "__main__":
	scatter_plot(sys.argv[1:])