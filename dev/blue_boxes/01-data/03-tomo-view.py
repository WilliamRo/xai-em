import os
import numpy as np



# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
folder_path = r'../../../data/Tomo110/tomo'

objects, labels = [], []
for fn in ('even.npy', 'odd.npy'):
  file_path = os.path.join(folder_path, fn)
  assert os.path.exists(file_path)
  x = np.load(file_path)
  objects.append(x)
  labels.append(fn)


# -----------------------------------------------------------------------------
# Show data in Omma
# -----------------------------------------------------------------------------
from pictor.plotters.microscope import Microscope
from xem.ui.omma import Omma

om = Omma('Omma', figure_size=(8, 8))
om.set_large_image(objects)
om.labels = labels

ms = om.add_plotter(Microscope())
ms.set('title', True)
ms.set('color_bar', True)
ms.set('mini_map', True)
ms.set('share_roi', True)
ms.zoom(0.5)
ms.sv(-3, 3)
om.sd(100)

om.show()


