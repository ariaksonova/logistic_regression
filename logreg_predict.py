import json
import sys
from argparse import ArgumentParser
from csv import DictWriter

import pandas as pd

# from .common.read_data import *


def parse_args(args, usage):
    parser = ArgumentParser(usage=usage)
    parser.add_argument('file_path',
                        type=str,
                        help='path to csv file')
    parser.add_argument('-s',
                        type=str,
                        help='output file name',
                        default='result.csv')
    parser.add_argument('-w',
                        type=str,
                        help='weights file',
                        default=const.SAVE_PATH)
    args = parser.parse_args(args)
    return args.file_path, args.s, args.w


def load_weights(file_path):
    with open(file_path, 'r') as fp:
        return json.loads(fp.read())


def save_to_csv(res, file_name):
    with open(file_name, 'w') as fp:
        fieldnames = ['Index', 'Hogwarts House']
        writer = DictWriter(fp, fieldnames)
        writer.writeheader()
        writer.writerows(res)


def predict(data, weights):
    res = []
    for i, row in data.iterrows():
        row_dict = row.to_dict()
        house_number = get_best_prediction(row_dict, weights)
        res.append(
            {'Index': int(row['Index']),
             'Hogwarts House': list(const.HOUSES.items())[house_number][0]})
    return res


def logreg_predict(args):
    try:
        file_path, save_file, weights_file = parse_args(
            args,
            'dslr logreg_predict [-h] [-s S] [-w W] file_path')
        data = pd.read_csv(
            file_path,
            skipinitialspace=True,
            usecols=const.INDEPENDENT_VARS + [const.HOUSE_COL, 'Index'])
        data = normalize_input(data)
        weights = load_weights(weights_file)
        res = predict(data, weights)
        save_to_csv(res, save_file)
        print('Done!')
        print('Result has been saved to "result.csv"')
    except FileNotFoundError:
        print('No weights found. Train model first.')
    except Exception:
        print('Wrong input file. Exiting...')


if __name__ == '__main__':
    """
    dslr logreg_predict [-h] [-s S] file_path
    """
    logreg_predict(sys.argv[1:])
