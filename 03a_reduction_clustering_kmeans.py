from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import scale

from util import getCreditCardData

# #############################################################################
# Visualize the results on PCA-reduced data
def visualeyes(reduced_data, clusterer, data_name, reduction_method, cluster_method):
    # reduced_data = PCA(n_components=2).fit_transform(data)
    # kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    clusterer.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = clusterer.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the {} dataset ({}-reduced data)\n'
            'Centroids are marked with white cross'.format(data_name, reduction_method))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('Figs/03_{}_clustering_on_{}_data_using_{}_reduction'.format(cluster_method, data_name, reduction_method))
    plt.clf()

def bench_k_means(estimator, name, X, y, data, labels, sample_size):
    t0 = time()
    if name == "LDA-based":
        estimator.fit(X, y)
    else:
        estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

    return [
            name,
            time() - t0, 
            estimator.inertia_,
            metrics.homogeneity_score(labels, estimator.labels_),
            metrics.completeness_score(labels, estimator.labels_),
            metrics.v_measure_score(labels, estimator.labels_),
            metrics.adjusted_rand_score(labels, estimator.labels_),
            metrics.adjusted_mutual_info_score(labels,  estimator.labels_, average_method='arithmetic'),
            metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=sample_size)
            ]

def run_k_means(X, y, data, labels, data_name, sample_size, n_components):
    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    m0 = bench_k_means(KMeans(init='k-means++', n_clusters=n_components, n_init=10),
                name="k-means++", X=X, y=y, data=data, labels=labels, sample_size=sample_size)

    bench_k_means(KMeans(init='random', n_clusters=n_components, n_init=10),
                name="random", X=X, y=y, data=data, labels=labels, sample_size=sample_size)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_components).fit(data)
    m1 = bench_k_means(KMeans(init=pca.components_, n_clusters=n_components, n_init=10),
                name="PCA-based",
                X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    # Visualize it
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=64, n_init=10)
    visualeyes(reduced_data, kmeans, data_name, 'PCA', 'K-means')

    ica = FastICA(n_components=n_components).fit(data)
    m2 = bench_k_means(KMeans(init=ica.components_, n_clusters=n_components, n_init=10),
                name="ICA-based",
                X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    # Visualize it
    reduced_data = FastICA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_components, n_init=10)
    visualeyes(reduced_data, kmeans, data_name, 'ICA', 'K-means')

    rtx = GaussianRandomProjection(n_components=n_components).fit(data)
    m3 = bench_k_means(KMeans(init=rtx.components_, n_clusters=n_components, n_init=1),
                name="RP-based",
                X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    # Visualize it
    reduced_data = GaussianRandomProjection(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_components, n_init=10)
    visualeyes(reduced_data, kmeans, data_name, 'Randomized Projection', 'K-means')

    nmf = FactorAnalysis(n_components=n_components).fit(data)
    m4 = bench_k_means(KMeans(init=nmf.components_, n_clusters=n_components, n_init=10),
                name="FactorAnalysis-based",
                X=X, y=y, data=data, labels=labels, sample_size=sample_size)
    # Visualize it
    reduced_data = FactorAnalysis(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_components, n_init=10)
    visualeyes(reduced_data, kmeans, data_name, 'FactorAnalysis', 'K-means')

    data = np.vstack((m0,m1,m2,m3,m4))
    df = pd.DataFrame(data, columns=['Method', 'Time', 'Inertia', 'Homogeneity', 'Completness', 'V-measure', 'ARI', 'AMI', 'Silhouette'])
    df.to_csv('Figs/03a_reduction_clustering_kmeans_{}.csv'.format(data_name))

    print(82 * '_')


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
run_k_means(X, y, data, labels, 'digits', sample_size, n_digits)

X, y, data = getCreditCardData('./Data/ccdefault.xls', subset=0.05)
n_samples, n_features = data.shape
n_outputs = len(np.unique(y))
# labels = data.columns.values
sample_size = 300
print("n_outputs: %d, \t n_samples %d, \t n_features %d"
      % (n_outputs, n_samples, n_features))
run_k_means(X, y, data, y, 'credit card', sample_size, 2)

