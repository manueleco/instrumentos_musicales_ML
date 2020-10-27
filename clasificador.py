import tensorflow as tf

import numpy as np
import IPython.display as display


#Cargar los TFRecords

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset