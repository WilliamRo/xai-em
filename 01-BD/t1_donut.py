import bd_core as core
import bd_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'donut'
id = 2
def model():
  from xem.layers.donut import Donut3D

  th = core.th
  model = m.get_initial_model()

  n_filter = int(th.archi_string.split('-')[0])
  model.add(Donut3D(n_filter, kernel_size=th.kernel_size,
                    pierce=th.int_para_1, verbose=True))
  m.mu.UNet(3, arc_string=th.archi_string).add_to(model)
  return m.finalize(model)


def main(_):
  console.start('{} on cryo-EM blind denoise task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'even2even'
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

  # Pierce dimension (comment the line below if not required)
  th.int_para_1 = 0
  assert th.int_para_1 in (0, None)

  th.kernel_size = 3
  th.activation = 'relu'

  # {filters}-{kernel_size}-{height}-{thickness}-[link_indices]-[mp]-[bn]
  th.archi_string = '8-3-3-2-relu-mp'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10
  th.early_stop = False
  th.probe_cycle = th.updates_per_round // 3

  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = True
  th.overwrite = False

  gif_mode = 1
  if gif_mode:
    th.epoch = 2
    th.probe_cycle = 1
    th.print_cycle = 1
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.mark += th.data_config.replace('>', '-')
  if th.int_para_1 is not None: th.mark += f'-pierce{th.int_para_1}'
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
