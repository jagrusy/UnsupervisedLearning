import sys
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.datasets import load_digits
from util import getCreditCardData, getWineData

def johnson_lindenstrauss(data, data_name):
    # `normed` is being deprecated in favor of `density` in histograms
    if LooseVersion(matplotlib.__version__) >= '2.1':
        density_param = {'density': True}
    else:
        density_param = {'normed': True}

    # Part 1: plot the theoretical dependency between n_components_min and
    # n_samples

    # range of admissible distortions
    eps_range = np.linspace(0.1, 0.99, 5)
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

    # range of number of samples (observation) to embed
    n_samples_range = np.logspace(1, 9, 9)

    plt.figure()
    for eps, color in zip(eps_range, colors):
        min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
        plt.loglog(n_samples_range, min_n_components, color=color)

    plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
    plt.xlabel("Number of observations to eps-embed")
    plt.ylabel("Minimum number of dimensions")
    plt.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components")
    plt.savefig('Figs/02b_rp_comp_samples')

    # range of admissible distortions
    eps_range = np.linspace(0.01, 0.99, 100)

    n_samples_range = np.logspace(2, 6, 5)
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

    plt.figure()
    for n_samples, color in zip(n_samples_range, colors):
        min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
        plt.semilogy(eps_range, min_n_components, color=color)

    plt.legend(["n_samples = %d" % n for n in n_samples_range], loc="upper right")
    plt.xlabel("Distortion eps")
    plt.ylabel("Minimum number of dimensions")
    plt.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")
    plt.savefig('Figs/02b_rp_comp_eps')

    # Part 2: perform sparse random projection of some digits images which are
    # quite low dimensional and dense or documents of the 20 newsgroups dataset
    # which is both high dimensional and sparse

    n_samples, n_features = data.shape
    print("Embedding %d samples with dim %d using various random projections"
        % (n_samples, n_features))

    n_components_range = np.array([1,10,100,1000])
    dists = euclidean_distances(data, squared=True).ravel()

    # select only non-identical samples pairs
    nonzero = dists != 0
    dists = dists[nonzero]

    for n_components in n_components_range:
        t0 = time()
        rp = SparseRandomProjection(n_components=n_components)
        projected_data = rp.fit_transform(data)
        print("Projected %d samples from %d to %d in %0.3fs"
            % (n_samples, n_features, n_components, time() - t0))
        if hasattr(rp, 'components_'):
            n_bytes = rp.components_.data.nbytes
            n_bytes += rp.components_.indices.nbytes
            print("Random matrix with size: %0.3fMB" % (n_bytes / 1e6))

        projected_dists = euclidean_distances(
            projected_data, squared=True).ravel()[nonzero]

        plt.figure()
        plt.hexbin(dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu)
        plt.xlabel("Pairwise squared distances in original space")
        plt.ylabel("Pairwise squared distances in projected space")
        plt.title("Pairwise distances distribution for n_components=%d" %
                n_components)
        cb = plt.colorbar()
        cb.set_label('Sample pairs counts')

        rates = projected_dists / dists
        print("Mean distances rate: %0.2f (%0.2f)"
            % (np.mean(rates), np.std(rates)))
        plt.savefig('Figs/02b_rp_pwdist_{}_{}'.format(data_name, n_components))

        plt.figure()
        plt.hist(rates, bins=50, range=(0., 2.), edgecolor='k', **density_param)
        plt.xlabel("Squared distances rate: projected / original")
        plt.ylabel("Distribution of samples pairs")
        plt.title("Histogram of pairwise distance rates for n_components=%d" %
                n_components)
        plt.savefig('Figs/02b_rp_histogram_{}_{}'.format(data_name, n_components))
        plt.clf()
        # TODO: compute the expected value of eps and add them to the previous plot
        # as vertical lines / region
 
np.random.seed(0)

digits = load_digits()
X, y = digits.data, digits.target
johnson_lindenstrauss(digits.data, 'digits')

X, y, data = getCreditCardData('./Data/ccdefault.xls', subset=0.2)
johnson_lindenstrauss(data, 'credit')