from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import scale

from util import getCreditCardData

def bench_gmm(estimator, name, X, y, data, labels, sample_size):
    t0 = time()
    estimator.fit(data)
    y_pred = estimator.predict(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.lower_bound_,
             metrics.homogeneity_score(labels, y_pred),
             metrics.completeness_score(labels, y_pred),
             metrics.v_measure_score(labels, y_pred),
             metrics.adjusted_rand_score(labels, y_pred),
             metrics.adjusted_mutual_info_score(labels,  y_pred,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, y_pred,
                                      metric='euclidean',
                                      sample_size=sample_size)))

    return [
            name,
            time() - t0, 
            estimator.lower_bound_,
            metrics.homogeneity_score(labels, y_pred),
            metrics.completeness_score(labels, y_pred),
            metrics.v_measure_score(labels, y_pred),
            metrics.adjusted_rand_score(labels, y_pred),
            metrics.adjusted_mutual_info_score(labels,  y_pred, average_method='arithmetic'),
            metrics.silhouette_score(data, y_pred, metric='euclidean', sample_size=sample_size)
            ]

def run_gmm(X, y, data, labels, data_name, sample_size, n_components):
    print(82 * '_')
    print('init\t\ttime\tlowbd\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    gmm = GaussianMixture(n_components=n_components, n_init=10)

    m1 = bench_gmm(gmm, name="GMM alone", X=X, y=y, data=data, labels=labels, sample_size=sample_size)

    # in this case the seeding of the centers is deterministic, hence we run the
    # GaussianMixture algorithm only once with n_init=1
    # bench_gmm(gmm, name="PCA", X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    reduced_data = PCA(n_components=2).fit_transform(data)
    m2 = bench_gmm(gmm, name="PCA reduced", X=X, y=y, data=reduced_data, labels=labels, sample_size=sample_size)
    # visualeyes(reduced_data, gmm, data_name, 'PCA', 'GMM')

    # ica = FastICA(n_components=n_components).fit(data)
    # bench_gmm(gmm, name="ICA", X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    reduced_data = FastICA(n_components=2).fit_transform(data)
    m3 = bench_gmm(gmm, name="ICA reduced", X=X, y=y, data=reduced_data, labels=labels, sample_size=sample_size)

    # rtx = GaussianRandomProjection(n_components=n_components).fit(data)
    # bench_gmm(gmm, name="RP", X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    reduced_data = GaussianRandomProjection(n_components=2).fit_transform(data)
    m4 = bench_gmm(gmm, name="RP reduced", X=X, y=y, data=reduced_data, labels=labels, sample_size=sample_size)

    # fac = FactorAnalysis(n_components=n_components).fit(data)
    # bench_gmm(gmm, name="FactorAnalysis", X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    reduced_data = FactorAnalysis(n_components=2).fit_transform(data)
    m5 = bench_gmm(gmm, name="Factor Analysis reduced", X=X, y=y, data=reduced_data, labels=labels, sample_size=sample_size)
    print(82 * '_')

    data = np.vstack((m1,m2,m3,m4,m5))
    df = pd.DataFrame(data, columns=['Method', 'Time', 'Lower Bound', 'Homogeneity', 'Completness', 'V-measure', 'ARI', 'AMI', 'Silhouette'])
    df.to_csv('Figs/03b_reduction_clustering_EM_{}.csv'.format(data_name))

np.random.seed(0)

digits = load_digits()
data = scale(digits.data)
X, y = digits.data, digits.target
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target
sample_size = 300
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
run_gmm(X, y, data, labels, 'digits', sample_size, 10)

X, y, data = getCreditCardData('./Data/ccdefault.xls', subset=1.0)
n_samples, n_features = data.shape
n_outputs = len(np.unique(y))
# labels = data.columns.values
sample_size = 300
print("n_outputs: %d, \t n_samples %d, \t n_features %d"
      % (n_outputs, n_samples, n_features))
run_gmm(X, y, data, y, 'credit card', sample_size, 2)

