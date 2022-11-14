from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class XEMConfig(SmartTrainerHub):

  link_indices_str = Flag.string('a', 'U-Net link indices', is_key=None)

  @property
  def link_indices(self):
    if self.link_indices_str in ('a', 'all', '-', ''):
      return self.link_indices_str
    return [int(s) for s in self.link_indices_str.split(',')]



# New hub class inherited from SmartTrainerHub must be registered
XEMConfig.register()
