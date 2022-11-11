from tframe import console



def probe(trainer):
  from tframe.trainers.trainer import Trainer
  from bd.bd_config import BDConfig
  from bd.bd_set import BDSet

  # Sanity check
  th = trainer.th
  assert isinstance(trainer, Trainer) and isinstance(th, BDConfig)

  # Get indices from th
  train_set: BDSet = trainer.training_set

  # Take snapshot
  train_set.snapshot(trainer.model)

  return 'Snapshot saved to checkpoint folder'
