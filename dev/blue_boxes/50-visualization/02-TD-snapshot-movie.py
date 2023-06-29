from roma import finder
from pictor import Pictor
from pictor.plotters.retina import Retina

import imageio
import os
import numpy as np



# Read images
folder_path = r'E:\xem\02-TD\02-TOMO\01_unet\checkpoints\1129_unet(16-3-3-2-relu-mp-a)even2odd-ws128'

get_iter = lambda fp: int(fp.split('-')[0][4:])

file_names = finder.walk(
  folder_path, 'file', pattern='Iter*-*.png', return_basename=True)
file_names.sort(key=get_iter)

images = [imageio.imread(os.path.join(folder_path, fn)) for fn in file_names]
# images = [(im - np.mean(im)) / np.std(im)  for im in images]


# Visualize image stack using pictor
p = Pictor(figure_size=(6, 6))
r = p.add_plotter(Retina())
r.set('color_bar', False)
r.set('title', True)
p.objects = images
p.labels = file_names
p.show()