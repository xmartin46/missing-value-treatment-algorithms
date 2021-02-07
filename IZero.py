import numpy as np

class IZero:
    def __init__(self):
        pass

    def impute(self, dataset):
        dataset[np.where(np.isnan(dataset))] = 0
        return dataset
