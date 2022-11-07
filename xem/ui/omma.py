from pictor import Pictor
from pictor.objects.image.large_image import LargeImage

import numpy as np



class Omma(Pictor):

  DEPTH_KEY = 'DePtH'

  # region: Overwriting

  def _register_default_key_events(self):
    super(Omma, self)._register_default_key_events()

    # Remove close key
    self.shortcuts.library.pop('q')
    self.shortcuts.library.pop('Escape')
    self.shortcuts.register_key_event('Q', self.quit, 'Quit')

    # Use command `q` to quit
    self.q = self.quit

  # endregion: Overwriting

  # region: Public Methods

  def set_large_image(self, li):
    if isinstance(li, np.ndarray): li = [li]
    self.objects = li
    if LargeImage.get_im_dim(li[0]) == 2: return

    # Make sure 3-D images have the same depth
    for im in li: assert im.shape[0] == li[0].shape[0]

    # Create dimension for 3-D image stack
    self.set_to_axis(self.DEPTH_KEY, list(range(li[0].shape[0])))

    # Register shortcut for depth dimension
    self.shortcuts.register_key_event(
      'N', lambda: self.set_cursor(self.DEPTH_KEY, 1, refresh=True),
      'Depth +1')
    self.shortcuts.register_key_event(
      'P', lambda: self.set_cursor(self.DEPTH_KEY, -1, refresh=True),
      'Depth -1')

  def set_depth_cursor(self, i: int):
    self.set_cursor(self.DEPTH_KEY, cursor=i - 1, refresh=True)
  sd = set_depth_cursor

  # endregion: Public Methods


if __name__ == '__main__':
  om = Omma()
  om.show()
