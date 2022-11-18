import bd_core as core
import bd_mu as m

from tframe import console
from tframe import SaveMode
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'quan'
id = 4

def predict(self, data):
  th = core.th

  from tframe import Predictor, DataSet
  import numpy as np
  assert isinstance(self, Predictor) and isinstance(data, DataSet)
  assert th.force_mask

  console.show_status(f'Averaging over {th.sample_num} samples ...')
  pred = 0
  for i in range(th.sample_num):
    console.print_progress(i, total=th.sample_num)
    pred += Predictor.predict(self, data)

  return pred / th.sample_num


def model():
  from tframe.advanced.blind_denoise.quan import InputDropout

  th = core.th
  model = m.get_initial_model()

  # Add input mask
  input_dropout: InputDropout = model.add(
    InputDropout(th.dropout, force_mask=th.force_mask))

  # Add U-Net backbone
  m.mu.UNet(3, arc_string=th.archi_string,
            link_indices=th.link_indices).add_to(model)

  # Add last layer
  model.add(m.mu.HyperConv3D(filters=1, kernel_size=1))

  # Build model
  quantity = input_dropout.get_loss(th.loss_string)
  model.build(loss=quantity, metric=quantity)

  if th.force_mask: model.predict = lambda data: predict(model, data)

  return model


def main(_):
  console.start('{} on cryo-EM blind denoise task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = ['even2odd', 'even2even'][1]
  th.train_volume_size = 64

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.link_indices_str = 'a'

  th.kernel_size = 3
  th.filters = 8
  th.activation = 'relu'

  # {filters}-{kernel_size}-{height}-{thickness}-[link_indices]-[mp]-[bn]
  th.archi_string = f'{th.filters}-{th.kernel_size}-3-2-{th.activation}-mp'

  # Configure input-mask logic
  th.dropout = 0.9
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.early_stop = True
  th.save_mode = SaveMode.ON_RECORD
  th.probe_cycle = th.updates_per_round

  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = 0
  th.overwrite = 1
  th.force_mask = 0
  th.sample_num = 10

  gif_mode = 0
  if gif_mode:
    th.epoch = 4
    th.probe_cycle = 2
    th.print_cycle = 2
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(
    model_name, th.archi_string + '-' + th.link_indices_str)
  th.mark += f'dp{th.dropout}'
  th.mark += f'-{th.data_config}'
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
