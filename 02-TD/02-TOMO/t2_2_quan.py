import td_core as core
import td_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'quan'
id = 2

def predict(self, data):
  th = core.th

  from tframe import Predictor, DataSet, context
  import numpy as np
  assert isinstance(self, Predictor) and isinstance(data, DataSet)
  assert th.force_mask

  drop_mask = context.get_collection_by_key('quan')['drop_mask']

  console.show_status(f'Averaging over {th.sample_num} samples ...')
  preds, masks = [], []
  for i in range(th.sample_num):
    console.print_progress(i, total=th.sample_num)
    pred, mask = Predictor.evaluate(self, [self.output_tensor, drop_mask], data)

    preds.append(pred)
    masks.append(mask)

  # Generate prediction
  pred = np.sum(preds, axis=0)

  mask = np.sum(masks, axis=0)
  mask[mask == 0] = 1e-6

  # Snapshot and validation
  if th.train: return pred / th.sample_num

  # Use omma to visualize results
  from xem.ui.omma import Omma
  om = Omma('Omma', figure_size=(8, 8))

  # Prepare data_dict to visualize
  data_dict = {'input': data.features[0]}
  if th.visualize_mask:
    for i, m in enumerate(masks):
      data_dict[f'mask[{i+1}]'] = m[0]
      data_dict[f'pred[{i+1}]'] = preds[i][0]
    # data_dict['sum(masks)'] = mask[0]

  # Put predictions into dataset
  MAX_PREDS = 10
  update_dict = {}
  for i, p in enumerate(preds):
    if i + 1 in (1, 10, 50, 100, 200, 1000):
      pred = np.sum(preds[:i+1], axis=0)
      mask = np.sum(masks[:i+1], axis=0)
      mask[mask == 0] = 1e-6
      update_dict[f'pred/mask, {i+1} samples'] = (pred / mask)[0]

    if i >= MAX_PREDS: continue
    data_dict[f'pred[{i+1}]'] = p[0]

  data_dict.update(update_dict)

  cmap = [None, 'gray'][0]
  om.visualize(data_dict, cmap=cmap, share_roi=True)
  # End of evaluation
  assert False


def model():
  from tframe.advanced.blind_denoise.quan import InputDropout

  th = core.th
  model = m.get_initial_model()

  # Add input mask
  input_dropout: InputDropout = model.add(
    InputDropout(th.dropout, force_mask=th.force_mask, mask_size=th.mask_size))

  # Add U-Net backbone
  m.mu.UNet(2, arc_string=th.archi_string,
            link_indices=th.link_indices).add_to(model)

  # Add last layer
  model.add(m.mu.HyperConv2D(filters=1, kernel_size=1))

  # Build model
  quantity = input_dropout.get_loss(th.loss_string)
  model.build(loss=quantity, metric=quantity)

  if th.force_mask:
    model.predict = lambda data: predict(model, data)
    model.output_tensor = model.output_tensor * input_dropout.drop_mask

  return model


def main(_):
  console.start('{} on 2-D TOMO denoise task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'tomo even2even 100'
  th.win_size = 128

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
  th.dropout = 0.5
  th.mask_size = 16
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.early_stop = True
  th.patience = 5
  th.probe_cycle = th.updates_per_round

  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = 0
  th.overwrite = 1
  th.force_mask = 1
  th.sample_num = 10
  # ---------------------------------------------------------------------------
  # 3.5. visualization
  # ---------------------------------------------------------------------------
  gif_mode = 0
  if gif_mode:
    th.epoch = 25
    th.probe_cycle = th.updates_per_round // 2

  th.visualize_mask = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(
    model_name, th.archi_string + '-' + th.link_indices_str)
  th.mark += th.data_config.split(' ')[1]
  th.mark += f'-ws{th.win_size}'
  th.mark += f'dp{th.dropout}sz{th.mask_size}'
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
