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
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .EM import EM

class ESD:
    def __init__(self):
        pass

    def __conditionalMeanAndCov(self, GMM, probabilities, Xi):
        # https://sci-hub.st/10.1016/j.asoc.2019.01.022
        Mi = np.isnan(Xi)
        Oi = ~np.isnan(Xi)

        # Mean and covariance matrix of the missing components
        mus_missing = np.zeros((GMM.n_gaussians, np.sum(Mi)), dtype=float)
        covs_missing = np.zeros((GMM.n_gaussians, np.sum(Mi), np.sum(Mi)), dtype=float)

        for cluster in range(GMM.n_gaussians):
            mu_cluster = GMM.mus[cluster]
            cov_cluster = GMM.covs[cluster]

            # mu_i
            mu_m = mu_cluster[Mi]
            mu_o = mu_cluster[Oi]
            cov_mo = cov_cluster[Mi, :][:, Oi]
            cov_oo = cov_cluster[Oi, :][:, Oi]
            cov_oo_inverse = np.linalg.inv(cov_oo + 1e-6 * np.identity(cov_oo.shape[0]))

            aux = np.dot(cov_mo, np.dot(cov_oo_inverse, (Xi[Oi] - mu_o)[:,np.newaxis]))
            nan_count = np.sum(Mi)
            mus_missing[cluster] = mu_m + aux.reshape(1, nan_count)

            # cov_i
            aux = np.linalg.inv(cov_cluster + 1e-6 * np.identity(cov_cluster.shape[0]))
            covs_missing[cluster] = np.linalg.inv(aux[Mi, :][:, Mi] + 1e-6 * np.identity(aux[Mi, :][:, Mi].shape[0]))


        co = np.zeros((GMM.d, GMM.d), dtype=float)
        mu = np.zeros((GMM.d), dtype=float)

        for cluster in range(GMM.n_gaussians):
            mu[Oi] += probabilities[cluster] * Xi[Oi]
            mu[Mi] += probabilities[cluster] * mus_missing[cluster]

            ri = 0
            for row, b1 in enumerate(Mi):
                if b1:
                    ci = 0
                    for col, b2 in enumerate(Mi):
                        if b2:
                            co[row][col] += probabilities[cluster] * covs_missing[cluster][ri][ci]
                            ci += 1
                    ri += 1

        # co = np.zeros((GMM.n_gaussians, GMM.d, GMM.d), dtype=float)
        # mu = np.zeros((GMM.n_gaussians, GMM.d), dtype=float)
        #
        # for cluster in range(GMM.n_gaussians):
        #     mu[cluster, Oi] += Xi[Oi]
        #     mu[cluster, Mi] += mus_missing[cluster]
        #
        #     ri = 0
        #     for row, b1 in enumerate(Mi):
        #         if b1:
        #             ci = 0
        #             for col, b2 in enumerate(Mi):
        #                 if b2:
        #                     co[cluster][row][col] += covs_missing[cluster][ri][ci]
        #                     ci += 1
        #             ri += 1

        return mu, co

    def __multivariate_normal_PDF(self, v, mu, cov):
        N = len(v)
        G = 1 / ( (2 * np.pi)**(N/2) * (np.linalg.det(cov + 1e-6 * np.identity(cov.shape[0])))**0.5 )
        G *= np.exp( - (0.5 * (v - mu).T.dot(np.linalg.inv(cov + 1e-6 * np.identity(cov.shape[0]))).dot( (v - mu) ) ) )
        return G

    def __computeProbabilities(self, dataset, GMM):
        probabilities = np.zeros((GMM.n_gaussians, GMM.n), dtype=float)

        for cluster in range(GMM.n_gaussians):
            for i in range(GMM.n):
                mu_cluster = GMM.mus[cluster]
                cov_cluster = GMM.covs[cluster]

                nan_indexes = np.isnan(dataset[i])
                mu_o = mu_cluster[~nan_indexes]
                cov_oo = cov_cluster[~nan_indexes, :][:, ~nan_indexes]
                # probabilities[cluster][i] = multivariate_normal.pdf(dataset[i][~nan_indexes], mean=mu_o, cov=cov_oo + 1e-6 * np.identity(cov_oo.shape[0]), allow_singular=False) * GMM.priors[cluster]
                probabilities[cluster][i] = self.__multivariate_normal_PDF(dataset[i][~nan_indexes], mu_o, cov_oo) * GMM.priors[cluster]

        aux = probabilities.sum(axis=0)
        probabilities = probabilities/(aux + 1e-308)
        return probabilities.T

    def estimateDistances(self, dataset, GMM, meanCovMissing=None):

        P = np.zeros((GMM.n, GMM.n), dtype=float)

        conditional_mean = np.zeros((GMM.n, GMM.d), dtype=float)
        conditional_covariance = np.zeros((GMM.n, GMM.d, GMM.d), dtype=float)

        probabilities = self.__computeProbabilities(dataset, GMM)

        if meanCovMissing == None:
            for i in range(GMM.n):
                conditional_mean[i], conditional_covariance[i] = self.__conditionalMeanAndCov(GMM, probabilities[i], dataset[i])
        else:
            (conditional_mean, conditional_covariance) = meanCovMissing

        for i in range(GMM.n):
            for j in range(GMM.n):
                if i > j:
                    # conditional_means_i = conditional_mean[i]
                    # conditional_means_j = conditional_mean[j]
                    #
                    # conditional_covariance_i = conditional_covariance[i]
                    # conditional_covariance_j = conditional_covariance[j]
                    #
                    # x_new_i = np.zeros((GMM.d), dtype=float)
                    # x_new_j = np.zeros((GMM.d), dtype=float)
                    # co_new_i = np.zeros((GMM.d, GMM.d), dtype=float)
                    # co_new_j = np.zeros((GMM.d, GMM.d), dtype=float)
                    #
                    # for cluster in range(GMM.n_gaussians):
                    #     x_new_i += probabilities[i][cluster] * conditional_means_i[cluster]
                    #     x_new_j += probabilities[j][cluster] * conditional_means_j[cluster]
                    #
                    #     co_new_i += probabilities[i][cluster] * conditional_covariance_i[cluster]
                    #     co_new_j += probabilities[j][cluster] * conditional_covariance_j[cluster]
                    #
                    # distance = 0
                    # # distance2 = 0
                    # for d in range(GMM.d):
                    #     distance += (x_new_i[d] - x_new_j[d]) ** 2 + co_new_i[d][d] + co_new_j[d][d]
                    #
                    #     # m = (x_new_i[d] - x_new_j[d])
                    #     # s = (x_new_i[d] - x_new_j[d]) ** 2 + co_new_i[d][d] + co_new_j[d][d]
                    #     # v = s - (m ** 2)
                    #     # distance2 += (m ** 2) + v

                    # distance = 0
                    # for d in range(GMM.d):
                    #     distance += (conditional_mean[i][d] - conditional_mean[j][d]) ** 2 + conditional_covariance[i][d][d] + conditional_covariance[j][d][d]
                    
                    distance = np.sum((conditional_mean[i] - conditional_mean[j]) ** 2) + np.trace(conditional_covariance[i]) + np.trace(conditional_covariance[j])
                    P[i][j] = distance
                    P[j][i] = distance

        return P