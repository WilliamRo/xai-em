import numpy as np

from tframe import DataSet
from roma import console



class BDSet(DataSet):
  """This class is for storing one big 3-D volume of data used for blind
     denoise. Note that
     (1) self.data_dict['odd|even|all'].shape = [D, H, W]
     (2) self.features|targets.shape = [N, D, H, W]
  """

  # region: Properties

  @DataSet.property()
  def data_for_validation(self):
    """This property is determined by
    (1) th.val_volume_size and th.val_volume_depth
    (2) val_volume_anchor := {(a1, a2, a3)} (can be multiple)
    """
    from bd_core import th

    anchors = [[int(coord_str) for coord_str in anchor_str.split(',')]
               for anchor_str in th.val_volume_anchor.split(';')]

    features, targets = [], []
    for anchor in anchors:
      x, y = self._get_volume(th.val_volume_depth, th.val_volume_size, anchor)
      features.append(x)
      targets.append(y)

    return BDSet(features=np.stack(features, axis=0),
                 targets=np.stack(targets, axis=0), name='Val-Set')

  @DataSet.property()
  def as_parent_data_set(self):
    return DataSet(features=self.features, targets=self.targets,
                   name=self.name + '(DataSet)')

  # endregion: Properties

  # region: Data feeding

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    """Generate mini-batches to feed an FNN during training or prediction
    (1) for training, generate <th.update_per_round> mini-batches per epoch.
        Each batch contains <th.batch_size> volumes of shape (L, L, L, 1).
        Here L = <th.train_volume_size>.
    (2) if <is_training> is False, generate one batch containing one volume
        of shape (L, L, L, 1), here L = <th.val_volume_size>.
    """
    assert is_training
    from bd_core import th

    round_len = self.get_round_length(batch_size, training=is_training)

    # Generate batches
    for i in range(round_len):
      features, targets = [], []
      for _ in range(batch_size):
        x, y = self._get_volume(th.train_volume_size, th.train_volume_size)

        # Switch x, y if necessary
        if th.random_switch and np.random.choice([True, False]): x, y = y, x

        features.append(x)
        targets.append(y)

      yield BDSet(np.stack(features, axis=0),
                  np.stack(targets, axis=0), name=f'TrainSet[{i+1}]')

    # Clear dynamic_round_len
    self._clear_dynamic_round_len()

  # endregion: Data feeding

  # region: Private Methods

  def _get_volume(self, depth, size, anchor=None):
    assert len(self.features) == 1

    # Randomly generate an anchor if not provided
    if anchor is None:
      anchor = [np.random.randint(0, L - size + 1)
                for L in self.features.shape[1:4]]

    assert len(anchor) == 3
    a1, a2, a3 = anchor
    return [x[0, a1:a1+depth, a2:a2+size, a3:a3+size]
            for x in (self.features, self.targets)]

  # endregion: Private Methods

  # region: Overwriting

  def _check_data(self):
    pass

  # endregion: Overwriting

  # region: Report and visualization

  def report(self):
    console.show_info(f'Details of {self.name}:')
    for k, v in self.data_dict.items():
      console.supplement(f'{k} shape: {v.shape}', level=2)

  def visualize(self, show_raw_data=True):
    from xem.ui.omma import Omma

    # Common configs
    configs = dict(title=True, color_bar=True, mini_map=True, share_roi=True,
                   vsigma=3, init_depth=100, init_zoom=0.5)

    # Create data_dict accordingly
    if show_raw_data:
      data_dict = {k: v for k, v in self.data_dict.items()
                   if k not in (self.FEATURES, self.TARGETS)}
    else:
      data_dict = {'feature': self.features[0], 'target': self.targets[0]}
      configs.update(dict(mini_map=False, init_depth=1, init_zoom=1.0))

    # Visualize data using Omma
    Omma.visualize(data_dict, **configs)

  # endregion: Report and visualization

  # region: Validation and snapshot

  def evaluate_denoiser(self, model):
    from bd_core import th
    from xem.ui.omma import Omma
    from tframe import Predictor

    assert isinstance(model, Predictor)

    # Create data_dict and visualize using Omma
    val_set = self.data_for_validation.as_parent_data_set
    result = model.predict(val_set)
    data_dict = {'feature': val_set.features[0], 'target': val_set.targets[0],
                 'result': result[0]}

    # Visualize data using Omma
    Omma.visualize(data_dict, init_depth=th.snapshot_d_indices_list[0],
                   vsigma=3, share_roi=True)

  def snapshot(self, model):
    from bd_core import th
    from tframe import Predictor
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)

    val_set = self.data_for_validation

    # (1) Get denoised image (shape=[1, D, H, W, 1])
    denoised_image = model.predict(val_set.as_parent_data_set)

    # (2) Get metrics
    val_dict = model.validate_model(val_set)

    # (3) Save image
    for d in th.snapshot_d_indices_list:
      metric_str = '-'.join([f'{k}{v:.2f}' for k, v in val_dict.items()])
      fn = f'Depth{d}-Iter{model.counter}-{metric_str}.png'
      plt.imsave(os.path.join(model.agent.ckpt_dir, fn),
                 denoised_image[0, d, ..., 0])

  # endregion: Validation and snapshot



if __name__ == '__main__':
  # ++ Blue box for data -> model
  from bd_core import th
  from bd.bd_agent import BDAgent
  from tframe import console

  th.data_config = 'even2odd'

  # Load data
  ds: BDSet = BDAgent.load()

  # (1) during training
  console.section('Training')

  th.updates_per_round = 5
  th.train_volume_size = 64

  for batch in ds.gen_batches(batch_size=32, is_training=True): batch.report()

  # (2) visualize data for validation
  console.section('Validation')

  th.val_volume_depth = 160
  th.val_volume_size = 320
  th.val_volume_anchor = '20,350,272'

  val_set = ds.data_for_validation
  val_set.report()
  val_set.visualize(show_raw_data=False)
