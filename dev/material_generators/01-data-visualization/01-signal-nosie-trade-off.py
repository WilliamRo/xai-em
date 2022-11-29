from td_core import th
from td.td_agent import TDAgent

import importlib.util
import sys



mod_path = r'E:\xem\02-TD\02-TOMO\01_unet\checkpoints\1129_unet(16-3-3-2-relu-mp-a)even2odd-ws128_visu\1129_unet(16-3-3-2-relu-mp-a)even2odd-ws128_visu.py'

# Import task sheet
spec = importlib.util.spec_from_file_location('task_sheet', mod_path)
task_sheet = importlib.util.module_from_spec(spec)
sys.modules['task_sheet'] = task_sheet
spec.loader.exec_module(task_sheet)

# Configure
th.developer_code = 'block_activate'
task_sheet.main(0)
th.data_config = 'tomo even2odd 100'

# Load data
train_set, val_set = TDAgent.load()

# Load denoiser and denoise
model = th.model()
result = model.predict(val_set)


# Visualize
from xem.ui.omma import Omma
from pictor.plotters.microscope import Microscope

even = train_set.features[0]
odd = train_set.targets[0]
denoised = result[0]

om = Omma('Omma', figure_size=(7, 7))

# Get objects and labels
image_list, label_list = [], []
image_list.append(even)
label_list.append('input')
image_list.append(denoised)
label_list.append('reference')

image_list.append(even)
label_list.append('input')
image_list.append(odd)
label_list.append('reference')

image_list.append(even)
label_list.append('input')
image_list.append(even)
label_list.append('reference')

# Set objects and labels
om.set_large_image(image_list)
om.labels = label_list

# Set microscope arguments
ms = om.add_plotter(Microscope())
ms.set('share_roi', True)
ms.set('title', True)

om.show()
