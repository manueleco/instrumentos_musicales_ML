import tensorflow as tf

import librosa

import numpy as np
import IPython.display as display


#Cargar los TFRecords

filenames = ["TFRecords/nsynth-train.tfrecord"]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset