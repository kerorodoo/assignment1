
import numpy as np
from skimage import transform as tf

# estimate transformation parameters
src = np.array([0, 0, 10, 10, 20, 20]).reshape((3, 2))
dst = np.array([12, 14, 1, -20, 5, 5]).reshape((3, 2))

tform = tf.estimate_transform('similarity', src, dst)

print tform._matrix

print tform(src)
