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
class kNN:
    def __init__(self):
        self.dataset = None
        self.O = None
        self.n = None
        self.d = None

    def __computeDistances(self, dataset, conju, q=2):
        distanceMatrix = []

        for i in range(self.n):
            print(i)
            all = np.tile(dataset[i], (self.n, 1))

            distances = (all - dataset) ** 2
            aux = np.nansum(distances, axis=1)
            aux2 = np.sqrt(np.divide(aux, conju[i]))
            aux2 = np.nan_to_num(aux2, nan=99999)

            res = []
            for c, v in enumerate(aux2):
                res.append((v, c))
            # res = [(v, c) for c, v in enumerate(aux2) if not np.isnan(dataset[c][miss_index]) ]
            distanceMatrix.append(res)

        return distanceMatrix

    def impute(self, dataset, k=5, q=2):
        self.dataset = dataset
        self.n, self.d = dataset.shape

        M = []

        # Observation matrix
        #       True      , if attribute is observed
        #       False     , otherwise
        self.O = ~np.isnan(dataset)

        conju = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            aux = np.tile(self.O[i, :], (self.n, 1))
            conju[i] = np.sum(np.logical_and(aux, self.O), axis=1)

        allDistances = self.__computeDistances(dataset, conju)

        # For each instance with missing values
        for i in range(self.n):
            print(i)
            elem = np.copy(dataset[i])
            miss_indexes = [x for x, val in enumerate(self.O[i]) if not val]

            for miss_index in miss_indexes:
                distances = allDistances[i]
                distances = [value for counter, value in enumerate(distances) if self.O[counter][miss_index] and counter != i]
                distances.sort()
                distances = distances[:k]

                # Impute missing values
                sum = 0
                m = len(distances)

                for [_, other] in distances:
                    sum += dataset[other][miss_index]

                if m != 0:
                    elem[miss_index] = sum/m
                else:
                    elem[miss_index] = 0

            M.append(elem)

        return np.array(M)

class kNN_parallel:
    def __init__(self):
        self.dataset = None
        self.O = None
        self.n = None
        self.d = None

    def __computeDistances(self, dataset, conju, q=2):
        distanceMatrix = []

        for i in range(self.n):
            all = np.tile(dataset[i], (self.n, 1))

            distances = (all - dataset) ** 2
            aux = np.nansum(distances, axis=1)
            aux2 = np.sqrt(np.divide(aux, conju[i]))
            aux2 = np.nan_to_num(aux2, nan=99999)

            res = []
            for c, v in enumerate(aux2):
                res.append((v, c))
            # res = [(v, c) for c, v in enumerate(aux2) if not np.isnan(dataset[c][miss_index]) ]
            distanceMatrix.append(res)

        return distanceMatrix

    def __par(self, dataset, allDistances, i, k, queue):
        elem = np.copy(dataset[i])
        miss_indexes = [x for x, val in enumerate(self.O[i]) if not val]

        for miss_index in miss_indexes:
            distances = allDistances[i]
            distances = [value for counter, value in enumerate(distances) if self.O[counter][miss_index] and counter != i]
            distances.sort()
            distances = distances[:k]

            # Impute missing values
            sum = 0
            m = len(distances)

            for [_, other] in distances:
                sum += dataset[other][miss_index]

            if m != 0:
                elem[miss_index] = sum/m
            else:
                elem[miss_index] = 0

        queue.put((i, elem))

    def impute(self, dataset, k=5, q=2):
        self.dataset = dataset
        self.n, self.d = dataset.shape
        queue = multiprocessing.Queue()
        M = []

        # Observation matrix
        #       True      , if attribute is observed
        #       False     , otherwise
        self.O = ~np.isnan(dataset)

        conju = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            aux = np.tile(self.O[i, :], (self.n, 1))
            conju[i] = np.sum(np.logical_and(aux, self.O), axis=1)

        allDistances = self.__computeDistances(dataset, conju)

        # For each instance with missing values
        starttime = time.time()
        processes = []
        for i in range(self.n):
            p = multiprocessing.Process(target=self.__par, args=(dataset, allDistances, i, k, queue))
            processes.append(p)
            p.start()

        outp = []
        for i in range(self.n):
            outp.append(queue.get())

        for process in processes:
            process.join()

        outp.sort()
        return np.array([tup[1] for tup in outp])
