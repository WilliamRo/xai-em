from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class XEMConfig(SmartTrainerHub):

  link_indices_str = Flag.string('a', 'U-Net link indices', is_key=None)



# New hub class inherited from SmartTrainerHub must be registered
XEMConfig.register()
