import td_core as core
import td_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'prism'
id = 3
def model():
  th = core.th
  model = m.get_initial_model()

  model.add(m.mu.HyperConv2D(filters=th.filters, kernel_size=th.kernel_size))
  model.add(m.mu.Activation('lrelu'))
  model.add(m.mu.HyperConv2D(filters=th.filters, kernel_size=th.kernel_size))
  model.add(m.mu.Activation('lrelu'))
  model.add(m.mu.HyperConv2D(filters=1, kernel_size=th.kernel_size))

  # Build model
  model.build(loss=th.loss_string, metric=th.loss_string)
  return model


def main(_):
  console.start('{} on 2-D EMPIAR denoise task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = ['empiar even2odd', 'empiar even2even'][0]
  th.win_size = 800

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

  th.filters = 48
  th.kernel_size = 11
  th.activation = 'lrelu'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 6
  th.early_stop = False
  th.probe_cycle = th.updates_per_round // 5

  th.batch_size = 4

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = 1
  th.overwrite = True

  gif_mode = 0
  if gif_mode:
    th.epoch = 2
    th.probe_cycle = 1
    th.print_cycle = 1
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  arc_string = f'f{th.filters}ks{th.kernel_size}'
  th.mark = '{}({})'.format(model_name, arc_string)
  th.mark += f'-ws{th.win_size}'
  th.mark += th.data_config
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
