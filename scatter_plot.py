import sys

import matplotlib.pyplot as plt
import pandas as pd

def scatter_plot(args):
    try:
        # file_path = _parse_args(args, 'dslr scatter_plot [-h] file_path')
        data = pd.read_csv("resources/dataset_train.csv")._get_numeric_data()
        plt.scatter(data['Defense Against the Dark Arts'], data['Astronomy'])
        plt.xlabel('Defense Against the Dark Arts')
        plt.ylabel('Astronomy')
        plt.show()
    except Exception:
        print('Wrong input file. Exiting...')


if __name__ == '__main__':
    """
    dslr scatter_plot [-h] file_path
    """
    scatter_plot(sys.argv[1:])
