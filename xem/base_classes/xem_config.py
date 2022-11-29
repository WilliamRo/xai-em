from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class XEMConfig(SmartTrainerHub):

  link_indices_str = Flag.string('a', 'U-Net link indices', is_key=None)
  force_mask = Flag.boolean(False, 'Whether to force masking')
  random_switch = Flag.boolean(
    False, 'Whether to randomly switch feature and target during training',
    is_key=None)

  mask_size = Flag.integer(1, 'Quan-mask size', is_key=None)
  visualize_mask = Flag.boolean(False, 'Option to visualize mask')

  erosion = Flag.integer(0, 'erosion in quan', is_key=None)


  @property
  def link_indices(self):
    if self.link_indices_str in ('a', 'all', '-', ''):
      return self.link_indices_str
    return [int(s) for s in self.link_indices_str.split(',')]



# New hub class inherited from SmartTrainerHub must be registered
XEMConfig.register()
