"""
This file contains a list of paths to be inserted into sys.path so that
scripts in this project can be run properly outside IDEs such as PyCharm.
"""
folders = [
  'xai-kit',
  'xai-kit/tframe/talos',
  'xai-kit/pictor',
  'xai-kit/roma',
]


def get_xai_alfa_dir():
  import os
  return os.path.dirname(os.path.abspath(__file__))
