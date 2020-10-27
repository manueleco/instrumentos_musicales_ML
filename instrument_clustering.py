import librosa
import librosa.display
import IPython.display as ipd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import scipy
import glob
import os
from numba import jit, cuda
from numba import vectorize

import pandas as pd
import seaborn as sns

from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers
import keras
from keras.models import Sequential
import warnings

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ======= Ejecucion de Kmeans clustering ========


def kmeansBasico(the_features):
    kmeans = KMeans(
        init="random",
        n_clusters=6,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(the_features)

    print(kmeans.inertia_)
    print(kmeans.cluster_centers_)
    print(kmeans.n_iter_)

    print(kmeans.labels_)

    # Instantiate k-means and dbscan algorithms
    kmeans = KMeans(n_clusters=2)
    dbscan = DBSCAN(eps=0.3)

    # Fit the algorithms to the features
    kmeans.fit(the_features)
    dbscan.fit(the_features)

    # Compute the silhouette scores for each algorithm
    kmeans_silhouette = silhouette_score(
        scaled_features, kmeans.labels_).round(25)
    dbscan_silhouette = silhouette_score(
        scaled_features, dbscan.labels_).round(25)

    kmeans_silhouette
    dbscan_silhouette

    # Plot the data and cluster silhouette comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 6), sharex=True, sharey=True
    )
    fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
    fte_colors = {
        0: "#008fd5",
        1: "#fc4f30",
    }
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
    ax1.set_title(
        f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
    )

    # The dbscan plot
    db_colors = [fte_colors[label] for label in dbscan.labels_]
    ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
    ax2.set_title(
        f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
    )
    plt.show()


def kmeansElaborado(data, true_label_names):
    df1 = pd.DataFrame(data=data)
    # print(data[:5, :3])
    # print(true_label_names[:5])
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(true_label_names)
    # true_labels[:5]
    # print(label_encoder.classes_)

    n_clusters = len(label_encoder.classes_)
    preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=2, random_state=None)),
        ]
    )
    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=50,
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )

    pipe.fit(data)

    preprocessed_data = pipe["preprocessor"].transform(data)

    predicted_labels = pipe["clusterer"]["kmeans"].labels_



    silhouette_score(preprocessed_data, predicted_labels)
    adjusted_rand_score(true_labels, predicted_labels)

    pcadf = pd.DataFrame(
        pipe["preprocessor"].transform(data),
        columns=["Variable_1","Variable_2"],
    )

    pcadf["Cluster_generado"] = pipe["clusterer"]["kmeans"].labels_
    pcadf["Label correcto"] = label_encoder.inverse_transform(true_labels)

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(8, 8))

    scat = sns.scatterplot(
        "Variable_1",
        "Variable_2",
        s=50,
        data=pcadf,
        hue="Cluster_generado",
        style="Label correcto",
        palette="Set2",
    )

    scat.set_title(
        "Resultados de clustering de piezas de Fur Elise"
    )
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()

    cluster_map = pd.DataFrame()
    # cluster_map['data_index'] = np.argmin(data)
    cluster_map['cluster'] = pipe["clusterer"]["kmeans"].labels_

    # print(cluster_map[cluster_map.cluster == 1])

    # print(df1)


def main():
    le_file = "dataset.csv"
    los_labels = 'labels.csv'

    # ------ Extraccion con Pandas -------
    features = pd.read_csv(r'dataset.csv')
    features = pd.DataFrame(features)

    # print(features)

    data = np.genfromtxt(le_file, delimiter=",",
                         usecols=range(1, 27), skip_header=1)

    true_label_names = np.genfromtxt(los_labels, delimiter=",",
                                     usecols=(1,), skip_header=1, dtype="str")

    featuresFixed = features.iloc[:, 1:27]
    # print(featuresFixed)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(featuresFixed)
    # print(scaled_features)
    print(data)
    kmeansElaborado(data, true_label_names)


if __name__ == "__main__":
    main()

# =========================================== CODIGO AUXILIAR ===========================================
