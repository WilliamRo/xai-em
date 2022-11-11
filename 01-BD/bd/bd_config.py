from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class BDConfig(SmartTrainerHub):

  train_volume_size = Flag.integer(64, 'Volume size for training', is_key=None)

  val_volume_size = Flag.integer(320, 'Volume size for validation')
  val_volume_depth = Flag.integer(160, 'Volume size for validation')
  val_volume_anchor = Flag.string(
    '0,0,0', 'Anchor point of validation volume. Use `;` to split '
             'multi-volumes', is_key=None)
  snapshot_d_indices = Flag.string(
    '80', 'Snapshot depth, use `,` to split multiple values')

  random_switch = Flag.boolean(
    False, 'Whether to randomly switch feature and target during training',
    is_key=None)


  @property
  def snapshot_d_indices_list(self):
    return [int(i) for i in self.snapshot_d_indices.split(',')]



# New hub class inherited from SmartTrainerHub must be registered
BDConfig.register()
