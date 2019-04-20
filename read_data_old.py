import pandas
import math
from argparse import ArgumentParser

def get_dataset():
	dataset = pandas.read_csv('resources/dataset_train.csv', sep=',')
	return dataset


def _parse_args(args, usage):
    parser = ArgumentParser(usage=usage)
    parser.add_argument('file_path',
                        type=str,
                        help='path to csv file')
    return parser.parse_args(args).file_path


def digitize_hand(row):
    row['Best Hand'] = const.HANDS.get(row['Best Hand'], 0)
    return row


def normalize_input(data, process_house=False):
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


def h(vars, weights):
    res = weights['bias']
    for key in const.INDEPENDENT_VARS:
        res += vars[key] * weights[key]
    return 1 / (1 + math.exp((-res)))


def get_best_prediction(vars, weights):
    res = []
    for weight in weights:
        res.append(h(vars, weight))
    return res.index(max(res))
