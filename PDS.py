import math
import copy
import numpy as np

class PDS():
    def __init__(self):
        pass

    def __computeDistances(self, dataset, conju, q=2):
        distanceMatrix = []

        for i in range(self.n):
            print(i)
            all = np.tile(dataset[i], (self.n, 1))

            distances = (all - dataset) ** 2
            aux = np.nansum(distances, axis=1)
            aux2 = np.multiply(aux, conju[i])
            distanceMatrix.append(aux2)

        return np.array(distanceMatrix)

    def estimateDistances(self, dataset):
        self.dataset = dataset
        self.n, self.d = dataset.shape

        # Observation matrix
        #       True      , if attribute is NOT observed
        #       False     , otherwise
        self.O = np.isnan(dataset)

        conju = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            aux = np.tile(self.O[i, :], (self.n, 1))
            with np.errstate(divide='ignore'):
                conju[i] = self.d/(self.d - np.sum(np.logical_or(aux, self.O), axis=1))
            conju[i][conju[i] == float("inf")] = 0

        distanceMatrix = self.__computeDistances(dataset, conju)

        originalMatrix = copy.deepcopy(distanceMatrix)

        mn = np.mean(distanceMatrix[np.nonzero(distanceMatrix)])
        distanceMatrix[distanceMatrix == 0] = mn
        np.fill_diagonal(distanceMatrix, 0)

        return originalMatrix, distanceMatrix
