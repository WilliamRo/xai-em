from xem.base_classes.xem_config import XEMConfig
from tframe.configs.config_base import Flag

from roma import Arguments



class TDConfig(XEMConfig):

  win_size = Flag.integer(None, 'Size of random window', is_key=None)

  @property
  def data_name(self):
    return Arguments.parse(self.data_config).func_name

  @property
  def data_kwargs(self) -> dict:
    return Arguments.parse(self.data_config).arg_dict



# New hub class inherited from SmartTrainerHub must be registered
TDConfig.register()
