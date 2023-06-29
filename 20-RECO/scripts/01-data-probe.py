import numpy as np
import os


data_dir = r'../data'
file_names = ['initial_data.npy', 'refine_data.npy']

KEYS = ('3d', '2d', 'eular')
def load_gordon_npy(fn):
  path = os.path.join(data_dir, fn)
  data_dict = np.load(path, allow_pickle=True).item()
  return [data_dict[k] for k in KEYS]

# shapes are: [64, 64, 64], [5812, 64, 64], [5812, 3]
init_model, init_candidates, init_euler = load_gordon_npy(file_names[0])
# shapes are: [256, 256, 256], [4748, 256, 256], [4748, 3]
refine_model, refine_candidates, refine_euler = load_gordon_npy(file_names[1])


