from collections import OrderedDict
from roma import console
from roma import finder
from td.td_set import TDSet

import os
import numpy as np
import tframe.utils.file_tools.io_utils as io_utils



class Tomo2DSet(TDSet):

  # region: Properties

  @TDSet.property()
  def validation_set(self):
    return self

  # endregion: Properties

  # region: Interfaces

  def configure(self):
    for key in ['features', 'targets']:
      self.data_dict[key] = self.data_dict[key][:, :, :1200]
    return self, self

  @classmethod
  def load_as_tframe_data(cls, data_root, **kwargs):
    """th.data_config example: tomo even2even 100"""
    from td_core import th

    # Load even, odd, and all volumes as numpy arrays
    data_dir = os.path.join(data_root, r'Tomo110/tomo')

    section_dict: OrderedDict = cls.load_as_numpy_array(data_dir)

    # Get 2D section from specified data
    data_dict = {}
    for data_key, cfg_key in zip(
        ['features', 'targets'], th.data_args[0].split('2')):
      assert cfg_key in ('even', 'odd', 'all')
      data_dict[data_key] = section_dict[cfg_key]

    return Tomo2DSet(data_dict=data_dict, name=f'Tomo2D[{th.data_args[1]}]')

  @classmethod
  def load_as_numpy_array(cls, data_dir) -> OrderedDict:
    from td_core import th

    depth = int(th.data_args[1])

    # If .section file exists, load it directly
    file_path = os.path.join(data_dir, f'tomo2D-{depth}.section')
    if os.path.exists(file_path): return io_utils.load(file_path)

    # Get sections from tomo volume, save, and return
    from bd.bd_agent import BDAgent

    volume_dict: OrderedDict = BDAgent.load_as_numpy_arrays(data_dir)
    data_dict = OrderedDict()
    for key in ('even', 'odd', 'all'):
      data_dict[key] = volume_dict[key][np.newaxis, depth]

    io_utils.save(data_dict, file_path)
    console.show_status(f'Data saved to `{file_path}`.')
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

      yield Tomo2DSet(np.stack(features, axis=0),
                      np.stack(targets, axis=0), name=f'TrainSet[{i+1}]')

    # Clear dynamic_round_len
    self._clear_dynamic_round_len()

  # endregion: Data feeding

  # region: Report and visualization

  def visualize(self, **kwargs):
    from xem.ui.omma import Omma

    cmap = [None, 'gray'][0]
    Omma.visualize({k: v[0] for k, v in self.data_dict.items()},
                   title=True, mini_map=True, share_roi=True,
                   cmap=cmap, **kwargs)

  # endregion: Report and visualization



if __name__ == '__main__':
  # ++ Blue box
  from td_core import th

  th.data_config = 'tomo even2odd 100'

  ds = Tomo2DSet.load_as_tframe_data(th.data_dir)
  ds.report()
  ds.visualize()
