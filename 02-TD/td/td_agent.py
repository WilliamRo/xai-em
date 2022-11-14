from td.td_set import TDSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console

import os
import numpy as np



class TDAgent(DataAgent):

  @classmethod
  def load(cls) -> TDSet:
    from td_core import th

    data_set = cls.load_as_tframe_data(th.data_dir, th.data_name,
                                       **th.data_kwargs)
    data_set.configure()
    data_set.report()

    return data_set.configure()


  @classmethod
  def load_as_tframe_data(cls, data_dir, data_name, **kwargs) -> TDSet:

    if data_name == 'empiar':
      from td.data_sets.empiar import EMPIARSet as DataSetClass
    else: raise KeyError(f'!! Unknown data name {data_name}')

    return DataSetClass.load_as_tframe_data(data_dir)



if __name__ == '__main__':
  # ++ Blue box for data
  from td_core import th

  th.data_config = 'empiar'

  train_set, val_set = TDAgent.load()
  val_set.visualize()
