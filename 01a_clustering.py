import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import time

from sklearn.datasets import load_digits
from util import getCreditCardData, getWineData
from sklearn.preprocessing import scale

np.random.seed(0)

def clustering(X, y, data_name):
    k_means_NMI = []
    k_means_AMI = []
    k_means_score = []
    gas_mix_NMI = []
    gas_mix_AMI = []
    gas_mix_score = []
    k_means_timing = []
    gas_mix_timing = []

    clusters = range(1, 25)
    # clusters = [1,2,4,8,16]

    viz = KElbowVisualizer(KMeans(random_state=0), k=clusters)
    viz.fit(X)
    viz.show(outpath='Figs/01_kmeans_cluster_{}_elbow.png'.format(data_name))
    bestK = viz.elbow_value_
    plt.clf()

    viz = SilhouetteVisualizer(KMeans(n_clusters=bestK, random_state=0))
    viz.fit(X)
    viz.show(outpath='Figs/01_kmeans_cluster_{}_silhouette.png'.format(data_name))
    plt.clf()

    k_means = KMeans(random_state=0)
    gas_mix = GaussianMixture(random_state=0)

    for i in clusters:
        # Cluster vs Features
        k_means.set_params(n_clusters=i)
        gas_mix.set_params(n_components=i)
        
        t1 = time.time()
        k_means.fit(X)
        t2 = time.time()
        gas_mix.fit(X)
        t3 = time.time()
        
        y_km = k_means.predict(X)
        y_gm = gas_mix.predict(X)

        k_means_NMI.append(normalized_mutual_info_score(y, y_km, average_method='arithmetic'))
        k_means_AMI.append(adjusted_mutual_info_score(y, y_km, average_method='arithmetic'))
        k_means_score.append(k_means.score(X))
        gas_mix_NMI.append(normalized_mutual_info_score(y, y_gm, average_method='arithmetic'))
        gas_mix_AMI.append(adjusted_mutual_info_score(y, y_gm, average_method='arithmetic'))
        gas_mix_score.append(gas_mix.score(X))
        k_means_timing.append(t2-t1)
        gas_mix_timing.append(t3-t2)

    df = pd.DataFrame()

    df.insert(loc=0, column='K-Means NMI', value=k_means_NMI)
    df.insert(loc=0, column='K-Means AMI', value=k_means_AMI)
    df.insert(loc=0, column='K-Means Score', value=k_means_score)
    df.insert(loc=0, column='Gaussian Mixture NMI', value=gas_mix_NMI)
    df.insert(loc=0, column='Gaussian Mixture AMI', value=gas_mix_AMI)
    df.insert(loc=0, column='Gaussian Mixture Score', value=gas_mix_score)
    df.insert(loc=0, column='K-Means Timing', value=k_means_timing)
    df.insert(loc=0, column='Gaussian Mixture Timing', value=gas_mix_timing)
    df.insert(loc=0, column='N Clusters', value=clusters)

    print(df.head())
    df.to_csv(path_or_buf='Figs/01_clustering_{}.csv'.format(data_name))


digits = load_digits()
X, y = digits.data, digits.target
clustering(scale(X), y, 'digits')

X, y, data = getCreditCardData('./Data/ccdefault.xls')
clustering(scale(X), y, 'credit')

