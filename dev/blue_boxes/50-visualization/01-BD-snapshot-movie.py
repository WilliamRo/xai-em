from roma import finder
from pictor import Pictor
from pictor.plotters.retina import Retina

import imageio
import os



# Read images
folder_path = r'E:\xem\01-BD\01_unet\checkpoints\1114_unet(8-3-3-2-relu-mp-1,2)even-even'

get_iter = lambda fp: int(fp.split('-')[1][4:])

file_names = finder.walk(
  folder_path, 'file', pattern='Depth*-Iter*-*.png', return_basename=True)
file_names.sort(key=get_iter)

images = [imageio.v2.imread(os.path.join(folder_path, fn)) for fn in file_names]


# Visualize image stack using pictor
p = Pictor(figure_size=(6, 6))
r = p.add_plotter(Retina())
r.set('color_bar', True)
r.set('title', True)
p.objects = images
p.labels = file_names
p.show()