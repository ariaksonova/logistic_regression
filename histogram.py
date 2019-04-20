import sys
import matplotlib.pyplot as plt

from parse_data import *

def histogram(argv):
	try:
		dataset = read_data(argv, "python3 histogram.py [-h] path_to_csv_file")
		dataset['Birthday'] = dataset['Birthday'].astype('datetime64[ns]')
		plt.hist([])
		plt.show()
	except Exception:
		print("Hmmmmmm... Something went wrong...")

if __name__ == "__main__":
	histogram(sys.argv[1:])