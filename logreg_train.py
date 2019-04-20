import json
import sys
from argparse import ArgumentParser
from copy import deepcopy

import pandas as pd
from . import variables

def normalize_data(data, process_house=False):
    def digitize_house(row):
        row['Hogwarts House'] = const.HOUSES[row['Hogwarts House']]
        return row

    def normalize(df):
        result = df.copy()
        for feature_name in const.INDEPENDENT_VARS:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = ((df[feature_name] - min_value)
                                    / (max_value - min_value))
        return result

    if process_house:
        data = data.dropna()
    else:
        data[const.HOUSE_COL].fillna(0, inplace=True)
    if process_house:
        data = data.apply(digitize_house, axis=1)
        data[const.HOUSE_COL].fillna(0, inplace=True)
    data = normalize(data)
    return data

def init_weights():
    return dict(map(lambda x: (x, 0), const.INDEPENDENT_VARS + ['bias', ]))


def check_acc(old, new, acc):
    for key in old.keys():
        if abs(old[key] - new[key]) >= accuracy:
            return False
    return True


def update_weights(weights, data, lr, house_num):
    for i, row in data.iterrows():
        res = 0 if row[const.HOUSE_COL] != house_num else 1
        hipotesis = h(row.to_dict(), weights)
        for col_name in const.INDEPENDENT_VARS:
            weights[col_name] += (lr * (res - hipotesis)
                                  * hipotesis * (1 - hipotesis)
                                  * row[col_name])
        weights['bias'] += lr * (res - hipotesis)
    return weights


def house_learn(data, lr, x, iter, acc):
    weights = init_weights()
    for i in range(iter):
        new_weights = update_weights(deepcopy(weights), data, lr, x)
        if check_acc(weights, new_weights, acc):
            break
        weights = new_weights
    return weights


def train(data, it=25, l=0.2, acc=0.1):
    weights = []
    for x in range(4):
        weights.append(house_learn(data, lr, x, it,
                                       acc))
    return weights


def save_weights(weights, filepath):
    with open(filepath, 'w') as fp:
        fp.write(json.dumps(weights))


def estimate_error(data, weights):
    guess = 0
    len_data = data.shape[0]
    for i, row in data.iterrows():
        if get_best_prediction(row.to_dict(), weights) == row[const.HOUSE_COL]:
            guess += 1
    return guess / len_data * 100


def parse_args(args, usage):
    parser = ArgumentParser(usage=usage)
    parser.add_argument('file_path',
                        type=str,
                        help='path to csv file')
    parser.add_argument('-o',
                        type=str,
                        help='output json file with weights',
                        default=const.SAVE_PATH)
    parser.add_argument('-n',
                        type=int,
                        help='number iterations',
                        default=25)
    parser.add_argument('-lr',
                        type=float,
                        help='learning rate',
                        default=0.2)
    parser.add_argument('-acc',
                        type=float,
                        help='accuracy',
                        default=0.1)
    args = parser.parse_args(args)
    return args.file_path, args.o, args.n, args.lr, args.acc


def logreg_train(args):
    try:
        # file_path, output_file, n, lr, acc = parse_args(
        #     args,
        #     'logreg_train.py [-h] [-o O] [-n N]'
        #     ' [-lr LR] [-acc ACC] file_path')
        file_path, output_file, n, lr, acc = "resources/dataset_train.csv", "result.json", 25, 0.2, 0.1
        data = pd.read_csv(
            file_path,
            skipinitialspace=True,
            usecols=variables.INDEPENDENT_VARS + [variables.HOUSE_COL, ])
        # data = normalize_data(data, process_house=True)
        # weights = train(data, num_iter=n, lr=lr, accuracy=acc)
        # save_weights(weights, output_file)
        # print('Model has been trained!')
        # print('Accuracy: {:3.2f}%'.format(estimate_error(data, weights)))
        # print(f'Weights has been saved to "{output_file}"')
    except Exception:
        print('Wrong input file. Exiting...')


if __name__ == '__main__':
    """
    dslr logreg_train [-h] [-o O] [-n N] [-lr LR] [-acc ACC] file_path
    """
    logreg_train(sys.argv[1:])
