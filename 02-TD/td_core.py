import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from tframe import Predictor

from tframe.trainers.smartrainer import SmartTrainerHub as Hub

import td_tu as du
import td_tu as tu


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()
th.data_dir = os.path.join(ROOT, 'data/Tomo110/tomo')

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.9

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
th.input_shape = [None, None, None, 1]

th.train_volume_size = 64

th.val_volume_size = 320
th.val_volume_depth = 160
th.val_volume_anchor = '20,350,272'
th.snapshot_d_indices = '80'
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.loss_string = 'mse'
th.early_stop = False
th.patience = 5

th.print_cycle = 2
th.updates_per_round = 50
th.validation_per_round = 1

th.export_tensors_upon_validation = True



def activate():
  # Load data
  data_set = du.load_data()

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Predictor)

  # Train or evaluate
  if th.train:
    model.train(data_set, validation_set=data_set.data_for_validation,
                probe=tu.probe, trainer_hub=th)
  else:
    data_set.evaluate_denoiser(model)

  # End
  model.shutdown()
  console.end()
