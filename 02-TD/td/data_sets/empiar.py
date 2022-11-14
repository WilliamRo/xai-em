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
    x = self.features[:, A:A+L, A:A+L]
    return EMPIARSet(x, x, name='EMPIAR-ValSet')

  # endregion: Properties

  # region: Interfaces

  def configure(self):
    mu, sigma = np.mean(self.features), np.std(self.features)
    self.features = (self.features - mu) / sigma

    return self, self.validation_set

  @classmethod
  def load_as_tframe_data(cls, data_root, **kwargs):
    large_image = cls.load_as_numpy_array(data_root)

    data_set = EMPIARSet(features=large_image, name='EMPIAR-1-raw')

    return data_set

  @classmethod
  def load_as_numpy_array(cls, data_root, **kwargs):
    """Returns only 1 large image"""
    from topaz.utils.data.loader import load_image

    data_dir = os.path.join(data_root, r'EMPIAR-10025\rawdata\micrographs')

    file_list = finder.walk(data_dir, pattern='*.mrc')
    file_path = file_list[0]

    first_image = np.array(load_image(file_path), copy=False).astype(np.float32)

    return first_image[np.newaxis, ..., np.newaxis]

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

        features.append(x)
        targets.append(x)

      yield EMPIARSet(np.stack(features, axis=0),
                      np.stack(targets, axis=0), name=f'TrainSet[{i+1}]')

    # Clear dynamic_round_len
    self._clear_dynamic_round_len()

  # endregion: Data feeding

  # region: Report and visualization

  def visualize(self):
    from xem.ui.omma import Omma

    Omma.visualize({'micrographs': self.features[0]},
                   title=False, vsigma=1, mini_map=True, init_zoom=0.5)

  # endregion: Report and visualization



if __name__ == '__main__':
  # ++ Blue box
  from td_core import th

  ds = EMPIARSet.load_as_tframe_data(th.data_dir)
  ds.report()
  ds.visualize()
