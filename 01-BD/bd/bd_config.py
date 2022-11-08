from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class BDConfig(SmartTrainerHub):

  volume_size = Flag.integer(64, 'Volume size', is_key=None)



# New hub class inherited from SmartTrainerHub must be registered
BDConfig.register()
