import pandas
import numpy as np
import math

def get_dataset():
    dataset = pandas.read_csv('resources/dataset_train.csv', sep=',')
    return dataset

def describe():
    dataset = get_dataset()
    print(dataset.columns)
    print(dataset["Flying"].tolist())

if __name__ == '__main__':
    describe()
