import argparse
import os
import pandas as pd
import numpy as np
from math import floor
from random import sample
from  matplotlib import pyplot as plt

class Knn:
    def __init__(self, data, knn, dist_metric):
        self.data = data
        self.knn = knn
        self.dist_metric = dist_metric

        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]
        self.dist_metrix = self.get_dist_metrix()


    def get_dist_metrix(self):
        '''
        vals is a m * n metrix
        cals is a m * m metrix
        m * n -> m * 1 * n -> m * m * n -> m * m
        '''

        vals =  self.X.values   # get the values of the dataframe (numpy)
        cals = self.get_dist_cal(vals)

        return (lambda v, i: pd.DataFrame(v, i, i))(cals, self.X.index) if cals is not None else None   # return as a dataframe


    def get_dist_cal(self, vals):
        result = None

        if self.dist_metric == 'euclidean':
            result = np.nansum((vals - vals[:, np.newaxis]) ** 2, axis=2) ** 0.5  # reshape metrix and compute the euclidean distance

        if self.dist_metric == 'manhattan':
            result = np.nansum(np.abs(vals - vals[:, np.newaxis]) , axis=2)  # reshape metrix and compute the manhattan distance

        return result


    def split(self, train_size_percent):
        size = len(self.data.index)

        train_idx = sample(range(size), floor(size * train_size_percent))   # randomly select records as training set
        test_idx = set(self.data.index) - set(train_idx)

        return self.X.loc[train_idx], self.X.loc[test_idx], self.y.loc[train_idx], self.y.loc[test_idx]


    def predict(self, train_X, test_X, train_y):
        result = []

        for idx, row in test_X.iterrows():
            distances = self.dist_metrix.iloc[idx][self.dist_metrix.columns.difference([idx])]  # retrieve distances between points (itself excluded)
            distances = distances.sort_values(ascending=True)

            neighbors = list(distances.index[:self.knn])    # get the k nearest neighbors based on the distances
            classes = train_y.loc[neighbors]    # get classes of selected neighbors

            result.append(classes.value_counts().idxmax())  # select the most frequent class as the prediction of test record

        return result

    def eval(self, test_y, predict_y, ground_truth):
        tp, fp, tn, fn = self.get_confusion_metrix(test_y, predict_y, ground_truth)

        result = pd.DataFrame(columns=['precision', 'recall', 'specificity', 'acc', 'f1'])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # the propotion of retrieved data that are relevant
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # the propotion of relevant data that successfully retrieved (so called true positive rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # so called true negative rate
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # the harmonic mean of precision and recall

        result.loc[len(result)] = [precision, recall, specificity, acc, f1]

        return result


    def get_confusion_metrix(self, test_y, predict_y, ground_truth):
        tp, tn, fp, fn = 0, 0, 0, 0

        for test, predict in zip(list(test_y), list(predict_y)):
            if test == predict and predict == ground_truth:
                tp += 1
            if test != predict and predict == ground_truth:
                fp += 1
            if test == predict and predict != ground_truth:
                tn += 1
            if test != predict and predict != ground_truth:
                fn += 1

        return tp, fp, tn, fn


    def plot(self, train_X, test_X, train_y, test_y, ground_truth):
        img_dir = 'image'

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        fig, ax = plt.subplots(figsize=(6, 4.5))

        cols = self.X.columns

        train_X.plot.scatter(x=cols[0], y=cols[1], edgecolor='black', marker='o', s=30, c=['black' if c == ground_truth else 'none' for c in list(train_y)], label='train', ax=ax)
        test_X.plot.scatter(x=cols[0], y=cols[1], edgecolor='black', marker='D', s=30, c=['black' if c == ground_truth else 'none' for c in list(test_y)], label='test', ax=ax)

        plt.title('KNN:' + str(self.knn))
        #plt.show()
        plt.savefig(os.path.join(img_dir, 'knn_{0}.png'.format(str(self.knn))))
        plt.close()



def float_range(val_min=None, val_max=None):
    def checker(val):
        try:
            val = float(val)

            if val_min is not None and val <= val_min:
                raise argparse.ArgumentTypeError('{0} must be > {1}'.format(val, val_min))

            if val_max is not None and val >= val_max:
                raise argparse.ArgumentTypeError('{0} must be < {1}'.format(val, val_max))
        except ValueError:
            raise argparse.ArgumentTypeError('{0} must be integer or float'.format(val))

        return val
    return checker


def main():
    parser = argparse.ArgumentParser()  # argument settings
    parser.add_argument('-d', '--dist_metric', choices=['euclidean', 'manhattan'], nargs='?', default='euclidean', help='distance between points in an n-dimensional space.')
    parser.add_argument('-f', '--filepath', nargs='?', default='sample.csv', help='csv file path.')
    parser.add_argument('-s', '--train_size_percent', type=float_range(0, 1), nargs='?', default=0.9, help='size of training set (percentage).')
    parser.add_argument('-k', '--knn', type=int, nargs='?', default=3, help='number of nearest neighbors')
    args = parser.parse_args()

    input_dir = 'input'
    ground_truth = 'Y'

    data = pd.read_csv(os.path.join(os.getcwd(), input_dir, args.filepath))
    print('\nData Set:')
    print(data)

    model = Knn(data, args.knn, args.dist_metric) # initialize knn classifier

    train_X, test_X, train_y, test_y = model.split(args.train_size_percent) # split into training set and test set
    print('\nPropotion of Training Set:\t' + str(args.train_size_percent))

    print('\nTraining Set:')
    print(train_X)

    print('\nTest Set:')
    print(test_X)

    predict_y = model.predict(train_X, test_X, train_y) # predict the class of test set
    seperator = ', '
    print('\nPredicted Classes:\t' + seperator.join(list(predict_y)))
    print('\nActual Classes:\t' + seperator.join(list(test_y)))

    evaluation = model.eval(test_y, predict_y, ground_truth) # performance evaluation of the knn classifier

    print('\nEvaluation:')
    print(evaluation)

    model.plot(train_X, test_X, train_y, test_y, ground_truth)


if __name__ == '__main__':
    main()
