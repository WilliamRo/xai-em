import bd_core as core
import bd_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'unet'
id = 1
def model():
  th = core.th

  return m.get_unet(th.archi_string, link_indices=th.link_indices)


def main(_):
  console.start('{} on cryo-EM blind denoise task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = ['even2odd', 'even2even'][1]
  th.random_switch = False
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

  th.link_indices_str = '1,2'

  th.kernel_size = 3
  th.activation = 'relu'

  # {filters}-{kernel_size}-{height}-{thickness}-[link_indices]-[mp]-[bn]
  th.archi_string = '8-3-3-2-relu-mp'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 20
  th.early_stop = False
  th.probe_cycle = th.updates_per_round // 2

  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = False
  th.overwrite = True

  gif_mode = 1
  if gif_mode:
    th.epoch = 4
    th.probe_cycle = 2
    th.print_cycle = 2
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(
    model_name, th.archi_string + '-' + th.link_indices_str)
  th.mark += th.data_config.replace('>', '-')
  if th.random_switch: th.mark += '-rs'
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
