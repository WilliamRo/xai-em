from tframe import console



def probe(trainer):
  from tframe.trainers.trainer import Trainer
  from td.td_config import TDConfig
  from td.td_set import TDSet

  # Sanity check
  th = trainer.th
  assert isinstance(trainer, Trainer) and isinstance(th, TDConfig)

  # Get indices from th
  val_set: TDSet = trainer.validation_set

  # Take snapshot
  val_set.snapshot(trainer.model)

  return 'Snapshot saved to checkpoint folder'
