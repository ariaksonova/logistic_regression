import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HANDS = {
    'Right': 0,
    'Left': 1
}


def digitize_hand(row):
    row['Best Hand'] = HANDS.get(row['Best Hand'], 0)
    return row

def pair_plot(args):
    try:
        # file_path = _parse_args(args, 'dslr pair_plot [-h] file_path')
        data = pd.read_csv("resources/dataset_train.csv")
        data = data.apply(digitize_hand, axis=1)
        data = data.dropna()
        g = sns.pairplot(data, hue='Hogwarts House',
                         palette={
                            'Gryffindor': 'r',
                            'Slytherin': 'g',
                            'Ravenclaw': 'b',
                            'Hufflepuff': 'y'
                         })
        handles = g._legend_data.values()
        labels = g._legend_data.keys()
        g.fig.legend(handles=handles, labels=labels, loc='lower center',
                     ncol=4)
        g.fig.subplots_adjust(top=0.99, bottom=0.05, left=0.05, right=0.95)
        plt.show()
    except Exception:
        print('Wrong input file. Exiting...')


if __name__ == '__main__':
    """
    dslr pair_plot [-h] file_path
    """
    pair_plot(sys.argv[1:])
