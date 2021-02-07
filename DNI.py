import numpy as np

class DNI:
    def __init__(self):
        pass

    def deleteElementsWithAnyNan(self, dataset):
        indices = np.array(np.where(np.any(np.isnan(np.array(dataset)), axis=1)))[0]
        dataset = np.delete(dataset, indices, axis=0)
        return dataset
