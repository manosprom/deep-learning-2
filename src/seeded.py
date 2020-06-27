import os
import random
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

seed = 19870127
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
