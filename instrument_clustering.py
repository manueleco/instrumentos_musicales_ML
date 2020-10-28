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

    dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]

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
        "Resultados de clustering de instrumentos musicales"
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
                                     usecols=(0,), skip_header=1, dtype="str")

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
