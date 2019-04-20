import sys
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style="ticks", color_codes=True)

from parse_data import *

def pair_plot(argv):
	try:
		dataset = read_data(argv, "python3 pair_plot.py [-h] path_to_csv_file")
		data_name = find_all_num(dataset)
		tmp_dataset = dataset[data_name]
		tmp_dataset = tmp_dataset.dropna()
		seaborn.pairplot(tmp_dataset, hue="Hogwarts House", markers = ".", height=2)
		plt.show()
	except Exception:
		print("Hmmmmmm... Something went wrong...")

if __name__ == "__main__":
	pair_plot(sys.argv[1:])