from typing import Tuple
from td.td_set import TDSet, DataSet
from td.td_agent import TDAgent



def load_data() -> Tuple[TDSet, TDSet]:
  train_set, val_set = TDAgent.load()

  train_set.report()
  val_set.report()
  return train_set, val_set



if __name__ == '__main__':
  from td_core import th

  th.data_config = 'empiar'

  train_set, val_set = load_data()
