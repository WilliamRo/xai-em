from bd.bd_set import BDSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console

import os
import numpy as np



class BDAgent(DataAgent):

  @classmethod
  def load(cls) -> BDSet:
    """Load blind denoise data

    th.data_config format
    ---------------------
    `feature`>`target`, e.g., even>odd, even,all>odd,all
    """
    from bd_core import th

    bd_set = cls.load_as_tframe_data(th.data_dir)

    # Parse data_config
    assert '>' in th.data_config
    for key, config in zip(
        (BDSet.FEATURES, BDSet.TARGETS), th.data_config.split('>')):
      volume_list = [bd_set.data_dict[k] for k in config.split(',')]

      # TODO: currently len(volume_list) is restricted to 1
      assert len(volume_list) == 1

      bd_set.data_dict[key] = np.stack(volume_list, axis=0)

    bd_set.name = f'BDSet({th.data_config})'
    bd_set.report()

    return bd_set


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> BDSet:
    """Since loading as numpy array is cheap, we don't save .tfd file here.
    """
    # Load even, odd, and all volumes as numpy arrays
    volume_dict: OrderedDict = cls.load_as_numpy_arrays(data_dir)

    # Wrap and return
    bd_set = BDSet(data_dict=volume_dict, name='BDSet(Raw)')
    return bd_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir) -> OrderedDict:
    console.show_status(f'Loading raw data from `{data_dir}` ...')

    volume_dict = OrderedDict()
    for key in ('even', 'odd', 'all'):
      file_path = os.path.join(data_dir, f'{key}.npy')
      if not os.path.exists(file_path): raise FileExistsError(
        f'!! `{file_path}` does not exist.')
      x = np.load(file_path)
      volume_dict[key] = np.expand_dims(x, -1)  # shape => (D, H, W, 1)

    return volume_dict



if __name__ == '__main__':
  # ++ Blue box for data
  from bd_core import th

  th.data_config = 'even>odd'
  ds = BDAgent.load()
  ds.data_dict['even+odd'] = ds['even'] + ds['odd']
  ds.report()
  ds.visualize()
