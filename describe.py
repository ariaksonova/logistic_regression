import pandas
import numpy as np
import math
import sys

from parse_data import *

class Describe:

    def __init__(self):
        self.name = []
        self.count = []
        self.mean = []
        self.std = []
        self.min = []
        self.quarter = []
        self.half = []
        self.three_quarter = []
        self.max = []

    def append_value(self, name, count, mean, std, min, quarter, half, three_quarter, max):
        self.name.append(name)
        self.count.append(count)
        self.mean.append(mean)
        self.std.append(std)
        self.min.append(min)
        self.quarter.append(quarter)
        self.half.append(half)
        self.three_quarter.append(three_quarter)
        self.max.append(max)

    def print_all(self):
        df = pandas.DataFrame()
        for i in range(len(self.name)):
            arr = []
            arr.append(self.count[i])
            arr.append(self.mean[i])
            arr.append(self.std[i])
            arr.append(self.min[i])
            arr.append(self.quarter[i])
            arr.append(self.half[i])
            arr.append(self.three_quarter[i])
            arr.append(self.max[i])
            df[self.name[i]] = arr
        df = df.rename({0: 'count', 1: 'mean', 2: 'std', 3: 'min', 4: '25%', 5: '50%', 6: '75%', 7: 'max'}, axis='index')
        print(df)

    def num_percentiles(self, n, percent):
        result = percent * (n - 1)
        return result

    def count_percent(self, arr, percent):
        num = self.num_percentiles(len(arr), percent)
        floor = math.floor(num)
        ceil = math.ceil(num)
        if floor == ceil:
            return arr[int(num)]
        result = (arr[int(floor)] * (ceil - num)) + (arr[int(ceil)] * (num - floor))
        return result

    def x_quarter(self, arr):
        arr.sort()
        x = self.count_percent(arr, 0.25)
        y = self.count_percent(arr, 0.50)
        z = self.count_percent(arr, 0.75)
        return x, y, z

    def x_min(self, arr):
        x_min = np.nan
        if len(arr) != 0:
            x_min = arr[0]
        for i in arr:
            if i < x_min:
                x_min = i
        return x_min

    def x_max(self, arr):
        x_max = np.nan
        if len(arr) != 0:
            x_max = arr[0]
        for i in arr:
            if i > x_max:
                x_max = i
        return x_max

    def x_mean(self, arr):
        sum = 0
        for i in arr:
            sum += i
        x_mean_value = sum / len(arr)
        return x_mean_value

    def x_std(self, arr):
        sum = 0
        x_mean_value = self.x_mean(arr)
        arr = [(x - x_mean_value) ** 2 for x in arr]
        for i in arr:
            sum += i
        return x_mean_value, math.sqrt(sum / (len(arr) - 1))

    def estimation(self, dataset, index):
        arr = [dataset.iloc[x][index] for x in range(len(dataset))]
        arr = list(filter(lambda i: str(i) != 'nan', arr))
        x, y, z = self.x_quarter(arr)
        x_mean_v, x_std_v = self.x_std(arr)
        self.append_value(dataset.columns[index], len(arr), x_mean_v, x_std_v, self.x_min(arr), x, y, z, self.x_max(arr))

def describe(argv):
    try:
        result = Describe()
        dataset = read_data(argv, "python3 describe.py [-h] path_to_csv_file")
        for i in range(len(dataset.columns)):
            if dataset.dtypes[i] == np.float64 or dataset.dtypes[i] == np.int64:
                result.estimation(dataset, i)
        result.print_all()
    except Exception:
        print("Hmmmmmm... Something went wrong...")

if __name__ == '__main__':
    describe(sys.argv[1:])
