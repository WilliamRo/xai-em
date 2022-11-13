from bd.bd_agent import BDAgent
from bd.bd_set import BDSet



load_data = BDAgent.load



if __name__ == '__main__':
  from bd_core import th

  th.data_config = 'even>odd'

  ds: BDSet = load_data()
  ds.report()