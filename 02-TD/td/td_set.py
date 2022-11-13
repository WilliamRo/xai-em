import numpy as np

from tframe import DataSet
from roma import console



class TDSet(DataSet):

  # region: Report and visualization

  def report(self):
    console.show_info(f'Details of {self.name}:')
    for k, v in self.data_dict.items():
      console.supplement(f'{k} shape: {v.shape}', level=2)

  def visualize(self):
    from pictor import Pictor
    from pictor.plotters import Retina

    p = Pictor('TD Data Viewer')
    p.add_plotter(Retina())

    img_shape = list(self.features.shape[1:])
    objects = np.stack([self.features, self.targets], axis=1).reshape(
      [2 * self.size] + img_shape)
    labels = []
    for i in range(self.size):
      labels.append(f'Pair[{i+1}] feature')
      labels.append(f'Pair[{i+1}] target')

    p.objects, p.labels = objects, labels
    p.show()

  # endregion: Report and visualization

  # region: Validation and snapshot

  def evaluate_denoiser(self, model):
    from td_core import th
    from tframe import Predictor

    assert isinstance(model, Predictor)

    pass

  def snapshot(self, model):
    from td_core import th
    from tframe import Predictor
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)

    # (1) Get denoised image (shape=[1, D, H, W, 1])
    denoised_image = model.predict(self)

    # (2) Get metrics
    val_dict = model.validate_model(self)

    # (3) Save image
    metric_str = '-'.join([f'{k}{v:.2f}' for k, v in val_dict.items()])
    fn = f'Iter{model.counter}-{metric_str}.png'
    plt.imsave(os.path.join(model.agent.ckpt_dir, fn), denoised_image[0])

  # endregion: Validation and snapshot



if __name__ == '__main__':
  # ++ Blue box for data -> model
  from td_core import th
  from bd.bd_agent import BDAgent
  from tframe import console

  th.data_config = 'even>odd'

  # Load data
  ds: BDSet = BDAgent.load()

  # (1) during training
  console.section('Training')

  th.updates_per_round = 5
  th.train_volume_size = 64

  for batch in ds.gen_batches(batch_size=32, is_training=True): batch.report()

  # (2) visualize data for validation
  console.section('Validation')

  th.val_volume_depth = 160
  th.val_volume_size = 320
  th.val_volume_anchor = '20,350,272'

  val_set = ds.data_for_validation
  val_set.report()
  val_set.visualize(show_raw_data=False)
