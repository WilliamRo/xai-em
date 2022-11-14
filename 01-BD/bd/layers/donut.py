from tframe import console
from tframe import tf
from tframe.layers.hyper.conv import ConvBase, Conv2D, Conv3D

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
    sub_shape = filter_shape[:-2]
    # Make sure all dims in sub_shape are odd integers
    for d in sub_shape: isinstance(d, int) and d % 2 == 1

    # Pierce if required
    p_index = self._nb_kwargs.get('pierce', None)
    if isinstance(p_index, int) and p_index >= 0:
      # Currently, this only works for piercing axis=0 in 3-D kernel
      assert p_index == 0 and self.Configs.kernel_dim == 3
      mask[:, sub_shape[1] // 2, sub_shape[2] // 2] = 0

      if self._nb_kwargs.get('verbose', False):
        console.show_info(f'Donut pierced (index = {p_index})')
    else:
      index = tuple([d // 2 for d in sub_shape])
      mask[index] = 0

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
