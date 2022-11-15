from collections import OrderedDict
from roma import console
from roma import finder
from td.td_set import TDSet

import os
import numpy as np



class EMPIARSet(TDSet):

  # region: Properties

  @TDSet.property()
  def validation_set(self):
    A, L = 1000, 2560
    x, y = self.features[:, A:A+L, A:A+L], self.targets[:, A:A+L, A:A+L]
    return EMPIARSet(x, y, name='EMPIAR-ValSet')

  # endregion: Properties

  # region: Interfaces

  def configure(self):
    from td_core import th

    for data_key, cfg_key in zip(
        ['features', 'targets'], th.data_args[0].split('2')):
      assert cfg_key in ('even', 'odd', 'all')
      self.data_dict[data_key] = self.data_dict[cfg_key]

    return self, self.validation_set

  @classmethod
  def load_as_tframe_data(cls, data_root, **kwargs):
    data_dict = cls.load_as_numpy_array(data_root)

    data_set = EMPIARSet(data_dict=data_dict, name='EMPIAR-1')

    return data_set

  @classmethod
  def load_as_numpy_array(cls, data_root, **kwargs) -> OrderedDict:
    data_dir = os.path.join(data_root, r'EMPIAR-10025/npy')
    console.show_status(f'Loading raw data from `{data_dir}` ...')

    data_dict = OrderedDict()

    for key in ('even', 'odd', 'all'):
      file_path = os.path.join(data_dir, f'{key}.npy')
      if not os.path.exists(file_path): raise FileExistsError(
        f'!! `{file_path}` does not exist.')
      x = np.load(file_path)

      # Normalize data
      x = (x - np.mean(x)) / np.std(x)

      data_dict[key] = x[np.newaxis, ..., np.newaxis]

    return data_dict

  # endregion: Interfaces

  # region: Data feeding

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    if not is_training:
      yield self.validation_set
      return

    from td_core import th

    round_len = self.get_round_length(batch_size, training=is_training)

    # Generate batches
    for i in range(round_len):
      features, targets = [], []
      for _ in range(batch_size):
        # Get random window
        size = th.win_size
        a1, a2 = [
          np.random.randint(0, L - size + 1) for L in self.features.shape[1:3]]
        x = self.features[0, a1:a1+size, a2:a2+size]
        y = self.targets[0, a1:a1+size, a2:a2+size]
        # Switch x, y if necessary
        if th.random_switch and np.random.choice([True, False]): x, y = y, x

        features.append(x)
        targets.append(y)

      yield EMPIARSet(np.stack(features, axis=0),
                      np.stack(targets, axis=0), name=f'TrainSet[{i+1}]')

    # Clear dynamic_round_len
    self._clear_dynamic_round_len()

  # endregion: Data feeding

  # region: Report and visualization

  def visualize(self):
    from xem.ui.omma import Omma

    Omma.visualize({k: v for k, v in self.data_dict.items()},
                   title=True, vsigma=1, mini_map=True, share_roi=True)

  # endregion: Report and visualization



if __name__ == '__main__':
  # ++ Blue box
  from td_core import th

  ds = EMPIARSet.load_as_tframe_data(th.data_dir)
  ds.report()
  ds.visualize()
