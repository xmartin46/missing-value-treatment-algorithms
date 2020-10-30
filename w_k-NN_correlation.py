import copy
import time
import math
import random
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ******************************** CLASSES ***********************************
class wNN_correlation:
    def __init__(self):
        self.dataset = None
        self.O = None
        self.n = None
        self.d = None
        self.cov = None

    def __C(self, r, m, c=0.5, type='Power'):
        if type == 'Power':
            return abs(r) ** m
        else:
            if abs(r) > c:
                return abs(r)/(1 - c) - c/(1 - c)
            else:
                return 0

    def __weight(self, dist, sum_distances, kernel_type, lambd):
        if sum_distances != 0:
            return self.__kernelize((dist/lambd), kernel_type)/sum_distances
        else:
            return 0

    def __kernelize(self, val, kernel='Gaussian'):
        if kernel == 'Gaussian':
            return (math.exp(-0.5 * (val ** 2)))/(math.sqrt(2 * math.pi))
        elif kernel == 'Tricube':
            return 70/85 * (1 - abs(val) ** 3) ** 3
        else:
            print("Any kernel selected")

    def __dM(self, dataset, conju, d):

        D = np.zeros((self.n, self.n), dtype=tuple)

        corrs = np.tile(self.cov[d], (self.n, 1))
        res = []

        for i in range(self.n):
            all = np.tile(dataset[i], (self.n, 1))

            diferences = (all - dataset) ** 2
            distances = np.multiply(diferences, corrs)
            aux = np.nansum(distances, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                aux2 = np.sqrt(np.divide(aux, conju[i]))
            aux2 = np.nan_to_num(aux2, nan=99999)

            D[i] = aux2

        return D

    def impute(self, real_dataset, dataset, k=5, q=2, kernel_type='Gaussian', lambd=5, mC=5):
        self.dataset = dataset
        self.n, self.d = dataset.shape
        # self.cov = self.__C(np.corrcoef(real_dataset.T), mC)
        self.cov = self.__C(np.array(pd.DataFrame(dataset).corr()), mC)
        queue = multiprocessing.Queue()

        # Observation matrix
        #       True      , if attribute is observed
        #       False     , otherwise
        self.O = ~np.isnan(dataset)

        conju = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            aux = np.tile(self.O[i, :], (self.n, 1))
            conju[i] = np.sum(np.logical_and(aux, self.O), axis=1)

        M = []

        copyDataset = copy.deepcopy(dataset)

        for d in range(self.d):
            print(d)
            allDistances_d = self.__dM(dataset, conju, d)

            for i in range(self.n):
                if not self.O[i][d]:
                    distances = allDistances_d[i]
                    distances = [(value, counter) for counter, value in enumerate(distances) if self.O[counter][d] and counter != i]
                    distances.sort()
                    distances = distances[:k]

                    # Impute missing values
                    sum_distances = 0
                    for [dist, _] in distances:
                        sum_distances += self.__kernelize((dist/lambd), kernel_type)

                    sum = 0
                    for [dist, other] in distances:
                        sum += self.__weight(dist, sum_distances, kernel_type, lambd) * dataset[other][d]

                    if sum != 0:
                        copyDataset[i][d] = sum
                    else:
                        copyDataset[i][d] = 0

        return copyDataset
