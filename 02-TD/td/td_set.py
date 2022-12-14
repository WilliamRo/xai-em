import numpy as np

from tframe import DataSet
from roma import console



class TDSet(DataSet):

  # region: Data IO

  def configure(self):
    pass

  @classmethod
  def load_as_numpy_array(cls, data_dir, **kwargs):
    raise NotImplementedError

  @classmethod
  def load_as_tframe_data(cls, data_dir, **kwargs):
    raise NotImplementedError

  # endregion: Data IO

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
    from xem.ui.omma import Omma
    from tframe import Predictor

    assert isinstance(model, Predictor)

    # This line is to protect predict method from going to customized
    # .. _gen_batches method
    self.__class__ = TDSet

    result = model.predict(self)

    data_dict = {'feature': self.features[0], 'target': self.targets[0],
                 f'Iter{model.counter}': result[0]}

    # Visualize data using Omma
    cmap = [None, 'gray'][0]
    Omma.visualize(data_dict, cmap=cmap, share_roi=True)


  def snapshot(self, model):
    from tframe import Predictor
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)

    # (1) Get denoised image (shape=[1, D, H, W, 1])
    denoised_images = model.predict(self)

    # (2) Get metrics
    val_dict = model.validate_model(self)

    # (3) Save image
    metric_str = '-'.join([f'{k}{v:.2f}' for k, v in val_dict.items()])
    fn = f'Iter{model.counter}-{metric_str}.png'
    plt.imsave(os.path.join(model.agent.ckpt_dir, fn),
               denoised_images[0, ..., 0])

  # endregion: Validation and snapshot



if __name__ == '__main__':
  pass

