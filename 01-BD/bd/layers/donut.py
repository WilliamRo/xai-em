from tframe import tf
from tframe.layers.hyper.conv import ConvBase, Conv2D, Conv3D

import typing as tp
import numpy as np



class DonutBase(ConvBase):

  def _get_filter(self, filter_shape):
    from tframe import hub as th

    # Get regularizer if necessary
    regularizer = None
    if th.use_global_regularizer: regularizer = th.get_global_regularizer()

    filter = tf.get_variable(
      name=self.abbreviation + 'weights', shape=filter_shape, dtype=th.dtype,
      initializer=self._weight_initializer, regularizer=regularizer)

    # Put a hole into filter
    mask = np.ones(shape=filter_shape)
    if self.Configs.kernel_dim == 1:
      mask[filter_shape[0] // 2] = 0
    elif self.Configs.kernel_dim == 2:
      mask[filter_shape[0] // 2, filter_shape[1] // 2] = 0
    else:
      mask[filter_shape[0] // 2, filter_shape[1] // 2, filter_shape[2] // 2] = 0

    return filter * mask


class Donut2D(Conv2D, DonutBase):

  full_name = 'donut2d'
  abbreviation = 'donut2d'

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    assert filter is None
    filter_shape = self._get_filter_shape(x.shape.as_list()[-1])

    return self.conv2d(
      x, self.channels, self.kernel_size, 'Donut2D', strides=self.strides,
      padding=self.padding, dilations=self.dilations,
      filter=self._get_filter(filter_shape), **kwargs)


class Donut3D(Conv3D, DonutBase):

  full_name = 'donut3d'
  abbreviation = 'donut3d'

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    assert filter is None
    filter_shape = self._get_filter_shape(x.shape.as_list()[-1])

    return self.conv3d(
      x, self.channels, self.kernel_size, 'Donut3D', strides=self.strides,
      padding=self.padding, dilations=self.dilations,
      filter=self._get_filter(filter_shape), **kwargs)
