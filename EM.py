import time
import math
import copy
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.random import normal
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# ******************************** CLASSES ***********************************
class EM_estimate:
    def __init__(self):
        self.n = None
        self.d = None
        self.n_gaussians = None
        self.priors = None
        self.mus = None
        self.covs = None

    def __initParameters(self, dataset, n_gaussians, init='kmeans'):
        priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
        mus = np.zeros((n_gaussians, self.d), dtype=float)
        covs = np.zeros((n_gaussians, self.d, self.d), dtype=float)

        cop = copy.deepcopy(dataset)
        # cop = np.where(np.isnan(cop), np.ma.array(cop, mask=np.isnan(cop)).mean(axis=0), cop)
        indices = np.array(np.where(np.all(~np.isnan(np.array(cop)), axis=1)))[0]
        if init == 'kmeans' and len(indices) > 0 and len(indices) > n_gaussians:
            print('KMEANS')
            data_for_kmeans = cop[indices]
            kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(data_for_kmeans)

            mus = kmeans.cluster_centers_

            if (np.all(np.array([len(data_for_kmeans[kmeans.labels_ == k, :]) > 1 for k in range(n_gaussians)]))):
                covs = np.array([np.cov(data_for_kmeans[kmeans.labels_ == k, :].T) for k in range(n_gaussians)])
                found = False
                for cluster in range(n_gaussians):
                    def is_pos_def(x):
                        return np.all(np.linalg.eigvals(x) > 0)
                    def check_symmetric(a, rtol=1e-05, atol=1e-08):
                        return np.allclose(a, a.T, rtol=rtol, atol=atol)

                    if not np.any(np.isnan(covs[cluster])):
                        if not is_pos_def(covs[cluster]) or not check_symmetric(covs[cluster]):
                            found = True
                    else:
                        found = True
            else:
                found = False
                covs = np.array([np.diag(np.nanvar(dataset, axis=0)) for _ in range(n_gaussians)])

            if found:
                covs = np.array([np.diag(np.nanvar(dataset, axis=0)) for _ in range(n_gaussians)])
                # covs = np.array([1e-6 * np.identity(self.d) for _ in range(n_gaussians])
                # covs = np.array([np.cov(mus[cluster]) for _ in range(n_gaussians])
                # covs = np.array([np.var(mus[cluster]) * np.identity(self.d) for _ in range(n_gaussians])
        # elif len(indices) > 0:
        #     print('NOT KMEANS BUT LEN(INDICES) > 0')
        #     for cluster in range(n_gaussians):
        #         mus[cluster] = dataset[indices[cluster]]
        #
        #     # FALTA COVS EN AQUEST CAS
        else:
            print('LEN(INDICES) <= 0')
            # impute with column means, just for initialization
            # randomly assign mean imputed dataset to K clusters
            impute = np.copy(dataset)
            col_mean = np.nanmean(impute, axis=0)
            inds = np.where(np.isnan(impute))
            impute[inds] = np.take(col_mean, inds[1])

            group = np.random.choice(n_gaussians, self.n)
            mus = np.array([np.mean(impute[group == k, :], axis=0) for k in range(n_gaussians)])
            covs = np.array([np.cov(impute[group == k, :].T) for k in range(n_gaussians)])

        return priors, mus, covs

    def __multivariate_normal_PDF(self, v, mu, cov):
        N = len(v)
        G = 1 / ( (2 * np.pi)**(N/2) * (np.linalg.det(cov))**0.5 )
        G *= np.exp( - (0.5 * (v - mu).T.dot(np.linalg.inv(cov)).dot( (v - mu) ) ) )
        return G

    def __e_step(self, dataset, priors, mus, covs, n_gaussians):
        probabilities = np.zeros((n_gaussians, self.n), dtype=float)

        for cluster in range(n_gaussians):
            probabilities[cluster] = priors[cluster] * multivariate_normal.pdf(dataset, mean=mus[cluster], cov=covs[cluster] + 1e-6 * np.identity(covs[cluster].shape[0]), allow_singular=False)
            # probabilities[cluster][i] = self.__multivariate_normal_PDF(dataset[i][~nan_indexes], mu_o, cov_oo + 1e-6 * np.identity(cov_oo.shape[0])) * priors[cluster]
        aux = probabilities.sum(axis=0)

        return probabilities/(aux + 1e-308), probabilities

    def __m_step(self, dataset, probabilities, priors, mus, covs, n_gaussians):
        for cluster in range(n_gaussians):
            mus[cluster] = (dataset * probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/(probabilities.sum(axis=1)[cluster] + 1e-308)
            temp3 = (dataset - mus[cluster]) * probabilities.T[:,cluster][:,np.newaxis]
            temp4 = (dataset - mus[cluster]).T
            covs[cluster] = np.dot(temp4, temp3)/(probabilities.sum(axis=1)[cluster] + 1e-308)

        priors = probabilities.sum(axis=1)/self.n
        return priors, mus, covs

    def estimate(self, dataset, n_gaussians, n_iters=50, epsilon=1e-8, init='kmeans', verbose=False):
        self.n, self.d = dataset.shape
        self.n_gaussians = n_gaussians

        priors, mus, covs = self.__initParameters(dataset, n_gaussians, init=init)

        actual = -float("Inf")

        it = 0
        while it < n_iters:
            probabilities, likelihood = self.__e_step(dataset, priors, mus, covs, n_gaussians)
            priors, mus, covs = self.__m_step(dataset, probabilities, priors, mus, covs, n_gaussians)

            next = np.sum(np.log(np.sum(likelihood.T, axis=1)))

            if verbose:
                print('logP(x) =', next)

            if next < actual:
                print('BUG!!!')
                break
            if np.abs(next - actual) < epsilon:
                print('Converged!')
                break

            temp = np.abs(next - actual)
            actual = next

            it += 1

        if it == n_iters - 1:
            print('Reached max. iterations!!')

        if verbose:
            print ("Iteration number = %d, stopping criterion = %.17f" %(it, temp))
            print("Final logP(x) => ", actual)

        self.priors = priors
        self.mus = mus
        self.covs = covs

        return priors, mus, covs, actual

    def MSIE(self, real, dataset):
        return super().MSIE(real, dataset)

    def MAIE(self, real, dataset):
        return super().MAIE(real, dataset)

class EM:
    def __init__(self):
        self.n = None
        self.d = None
        self.n_gaussians = None
        self.priors = None
        self.mus = None
        self.covs = None

    def __initParameters(self, dataset, n_gaussians, init='kmeans', verbose=False):
        priors = np.asarray(np.repeat(1/n_gaussians, n_gaussians), dtype=float)
        mus = np.zeros((n_gaussians, self.d), dtype=float)
        covs = np.zeros((n_gaussians, self.d, self.d), dtype=float)

        cop = copy.deepcopy(dataset)
        # cop = np.where(np.isnan(cop), np.ma.array(cop, mask=np.isnan(cop)).mean(axis=0), cop)
        indices = np.array(np.where(np.all(~np.isnan(np.array(cop)), axis=1)))[0]
        if init == 'kmeans' and len(indices) > 0 and len(indices) > n_gaussians:
            if verbose:
                print('KMEANS')
            data_for_kmeans = cop[indices]
            kmeans = KMeans(n_clusters=n_gaussians, init='k-means++').fit(data_for_kmeans)

            mus = kmeans.cluster_centers_

            # if (np.all(np.array([len(data_for_kmeans[kmeans.labels_ == k, :]) > 1 for k in range(n_gaussians)]))):
            #     covs = np.array([np.cov(data_for_kmeans[kmeans.labels_ == k, :].T) for k in range(n_gaussians)])
            #     found = False
            #     for cluster in range(n_gaussians):
            #         def is_pos_def(x):
            #             return np.all(np.linalg.eigvals(x) > 0)
            #         def check_symmetric(a, rtol=1e-05, atol=1e-08):
            #             return np.allclose(a, a.T, rtol=rtol, atol=atol)
            #
            #         if not np.any(np.isnan(covs[cluster])):
            #             if not is_pos_def(covs[cluster]) or not check_symmetric(covs[cluster]):
            #                 found = True
            #         else:
            #             found = True
            # else:
            #     found = False
            #     covs = np.array([np.diag(np.nanvar(dataset, axis=0)) for _ in range(n_gaussians)])
            #
            # if found:
            #     covs = np.array([np.diag(np.nanvar(dataset, axis=0)) for _ in range(n_gaussians)])
            #     # covs = np.array([1e-6 * np.identity(self.d) for _ in range(n_gaussians])
            #     # covs = np.array([np.cov(mus[cluster]) for _ in range(n_gaussians])
            #     # covs = np.array([np.var(mus[cluster]) * np.identity(self.d) for _ in range(n_gaussians])

            # PRIVILEGE MODE
            covs = np.array([np.diag(np.nanvar(dataset, axis=0)) for _ in range(n_gaussians)])
        # elif len(indices) > 0:
        #     print('NOT KMEANS BUT LEN(INDICES) > 0')
        #     for cluster in range(n_gaussians):
        #         mus[cluster] = dataset[indices[cluster]]
        #
        #     # FALTA COVS EN AQUEST CAS
        else:
            if verbose:
                print('LEN(INDICES) <= 0')
            # impute with column means, just for initialization
            # randomly assign mean imputed dataset to K clusters
            impute = np.copy(dataset)
            col_mean = np.nanmean(impute, axis=0)
            inds = np.where(np.isnan(impute))
            impute[inds] = np.take(col_mean, inds[1])

            group = np.random.choice(n_gaussians, self.n)
            mus = np.array([np.mean(impute[group == k, :], axis=0) for k in range(n_gaussians)])
            covs = np.array([np.cov(impute[group == k, :].T) for k in range(n_gaussians)])

        return priors, mus, covs

    def __C(self, dataset, probabilities, cov, cluster):
        C = np.zeros((self.d, self.d))

        cov_inv = np.linalg.inv(cov + 1e-6 * np.identity(cov.shape[0]))

        for i in range(self.n):
            nan_indexes = np.isnan(dataset[i])
            if np.any(nan_indexes):
                # cov_mm = cov[nan_indexes, :][:, nan_indexes]
                # cov_mo = cov[nan_indexes, :][:, ~nan_indexes]
                # cov_oo = cov[~nan_indexes, :][:, ~nan_indexes]
                # cov_oo_inverse = np.linalg.pinv(cov_oo)
                #
                # aux = cov_mm - np.dot(cov_mo, np.dot(cov_oo_inverse, cov_mo.T))

                aux = np.linalg.inv(cov_inv[nan_indexes, :][:, nan_indexes] + 1e-6 * np.identity(cov_inv[nan_indexes, :][:, nan_indexes].shape[0]))

                aux = (probabilities[cluster][i] / (probabilities.sum(axis=1)[cluster] + 1e-308)) * aux

                ri = 0
                for row, b1 in enumerate(nan_indexes):
                    if b1:
                        ci = 0
                        for col, b2 in enumerate(nan_indexes):
                            if b2:
                                C[row][col] += aux[ri][ci]
                                ci += 1
                        ri += 1

        return C

    def __multivariate_normal_PDF(self, v, mu, cov):
        N = len(v)
        G = 1 / ( (2 * np.pi)**(N/2) * (np.linalg.det(cov))**0.5 )
        G *= np.exp( - (0.5 * (v - mu).T.dot(np.linalg.inv(cov)).dot( (v - mu) ) ) )
        return G

    def __e_step(self, dataset, priors, mus, covs, n_gaussians):
        probabilities = np.zeros((n_gaussians, self.n), dtype=float)

        for cluster in range(n_gaussians):
            for i in range(self.n):
                mu_cluster = mus[cluster]
                cov_cluster = covs[cluster]

                nan_indexes = np.isnan(dataset[i])
                mu_o = mu_cluster[~nan_indexes]
                cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
                # probabilities[cluster][i] = multivariate_normal.pdf(dataset[i][~nan_indexes], mean=mu_o, cov=cov_oo + 1e-6 * np.identity(cov_oo.shape[0]), allow_singular=False) * priors[cluster]
                probabilities[cluster][i] = self.__multivariate_normal_PDF(dataset[i][~nan_indexes], mu_o, cov_oo + 1e-6 * np.identity(cov_oo.shape[0])) * priors[cluster]

        aux = probabilities.sum(axis=0)
        return probabilities/(aux + 1e-308), probabilities

    def __m_step(self, dataset, probabilities, priors, mus, covs, n_gaussians):

        for cluster in range(n_gaussians):
            # Expect missing values
            data_aux = copy.deepcopy(dataset)
            for i in range(self.n):
                if np.sum(np.isnan(dataset[i])) != 0:
                    mu_cluster = mus[cluster]
                    cov_cluster = covs[cluster]

                    nan_indexes = np.isnan(dataset[i])
                    mu_m = mu_cluster[nan_indexes]
                    mu_o = mu_cluster[~nan_indexes]
                    cov_mo = cov_cluster[nan_indexes, :][:, ~nan_indexes]
                    cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
                    cov_oo_inverse = np.linalg.inv(cov_oo + 1e-6 * np.identity(cov_oo.shape[0]))

                    aux = np.dot(cov_mo, np.dot(cov_oo_inverse, (dataset[i, ~nan_indexes] - mu_o)[:,np.newaxis]))
                    nan_count = np.sum(nan_indexes)
                    data_aux[i, nan_indexes] = mu_m + aux.reshape(1, nan_count)

            mus[cluster] = (data_aux * probabilities.T[:,cluster][:,np.newaxis]).sum(axis=0)/(probabilities.sum(axis=1)[cluster] + 1e-308)
            temp3 = (data_aux - mus[cluster]) * probabilities.T[:,cluster][:,np.newaxis]
            temp4 = (data_aux - mus[cluster]).T
            covs[cluster] = np.dot(temp4, temp3)/(probabilities.sum(axis=1)[cluster] + 1e-308) + self.__C(dataset, probabilities, covs[cluster], cluster)

        priors = probabilities.sum(axis=1)/self.n

        return priors, mus, covs

    def impute(self, dataset, n_gaussians, n_iters=50, epsilon=1e-8, init='kmeans', verbose=False):
        self.n, self.d = dataset.shape
        self.n_gaussians = n_gaussians

        priors, mus, covs = self.__initParameters(dataset, n_gaussians, init=init, verbose=verbose)

        actual = -float("Inf")

        it = 0
        while it < n_iters:
            probabilities, likelihood = self.__e_step(dataset, priors, mus, covs, n_gaussians)
            priors, mus, covs = self.__m_step(dataset, probabilities, priors, mus, covs, n_gaussians)

            next = np.sum(np.log(np.sum(likelihood.T, axis=1)))

            if verbose:
                print('logP(x) =', next)

            if next < actual:
                print('BUG!!!')
                break
            if np.abs(next - actual) < epsilon:
                print('Converged!')
                break

            temp = np.abs(next - actual)
            actual = next

            it += 1

        if it == n_iters:
            print('Reached max. iterations!!')

        if verbose:
            print ("Iteration number = %d, stopping criterion = %.17f" %(it, temp))
            print("Final logP(x) => ", actual)

        self.priors = priors
        self.mus = mus
        self.covs = covs

        return priors, mus, covs
