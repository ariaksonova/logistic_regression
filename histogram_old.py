import sys
import matplotlib.pyplot as plt
import pandas as pd
import math
from argparse import ArgumentParser

def histogram(args):
    try:
        # file_path = _parse_args(args, 'dslr histogram [-h] file_path')
        data = pd.read_csv("resources/dataset_train.csv")
        data['Birthday'] = data['Birthday'].astype('datetime64[ns]')
        course_column = data['Birthday'].apply(lambda x: x.year)
        data = data.assign(Course=course_column)
        bins = sorted(data.Course.unique())
        ravenclaw = data.loc[data['Hogwarts House'] == 'Ravenclaw']
        slytherin = data.loc[data['Hogwarts House'] == 'Slytherin']
        gryffindor = data.loc[data['Hogwarts House'] == 'Gryffindor']
        hufflepuff = data.loc[data['Hogwarts House'] == 'Hufflepuff']
        plt.hist([ravenclaw['Course'],
                  gryffindor['Course'],
                  hufflepuff['Course'],
                  slytherin['Course']],
                 bins=bins,
                 label=['Ravenclaw',
                        'Slytherin',
                        'Gryffindor',
                        'Hufflepuff'],
                 color=['b',
                        'g',
                        'r',
                        'y'])
        plt.legend(loc='upper right')
        plt.show()
    except Exception:
        print('Wrong input file. Exiting...')


if __name__ == '__main__':
    """
    dslr histogram [-h] file_path
    """
    histogram(sys.argv[1:])
