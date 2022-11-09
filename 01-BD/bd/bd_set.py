from tframe import DataSet
from pictor import Pictor
from roma import console



class BDSet(DataSet):
  """This class is for storing one big 3-D volume of data used for blind
     denoise. Note that
     (1) self.data_dict['odd|even|all'].shape = [D, H, W]
     (2) self.features|targets.shape = [N, D, H, W]
  """

  # region: Properties

  # endregion: Properties

  # region: Overwriting

  def _check_data(self):
    pass

  # endregion: Overwriting

  # region: Report and visualization

  def report(self):
    console.show_info(f'Details of {self.name}:')
    for k, v in self.data_dict.items():
      console.supplement(f'{k} shape: {v.shape}', level=2)

  def visualize(self):
    from xem.ui.omma import Omma

    # Visualize data except features and targets
    Omma.visualize(
      {k: v for k, v in self.data_dict.items()
       if k not in (self.FEATURES, self.TARGETS)},
      title=True, color_bar=True, mini_map=True, share_roi=True,
      vsigma=3, init_depth=100, init_zoom=0.5)

  # endregion: Report and visualization
