from tframe.advanced.blind_denoise.quan import InputDropout
from tframe import context, console, Predictor
from pictor import Pictor

import numpy as np
import tensorflow as tf


console.suppress_logging()
tf.InteractiveSession()

x = tf.constant(0, dtype=tf.float32, shape=[1, 64, 64, 1])
p = Pictor.image_viewer('Mask Viewer')

quan = InputDropout(mask_size=8)
quan.dim = 2
mask = quan._tf_gen_mask(x)
kernel = tf.constant(1.0, dtype=tf.float32, shape=[3, 3, 1])

strides, rates = [1] * 4, [1] * 4
eroded = tf.nn.erosion2d(mask, kernel=kernel, strides=strides, rates=rates,
                         padding='SAME')
eroded += 1

sess = tf.get_default_session()
values = sess.run([mask, eroded])

p.objects = [m[0] for m in values]
p.show()







