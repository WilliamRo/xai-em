from tframe.advanced.blind_denoise.quan import InputDropout
from tframe import console
from tframe import tf

from xem.ui.omma import Omma

from td_core import th
from td.td_agent import TDAgent


console.suppress_logging()
tf.InteractiveSession()
sess = tf.get_default_session()

# Load data
th.data_config = 'tomo even2odd 100'
train_set, _ = TDAgent.load()

features = train_set.features
tensor = tf.placeholder(dtype=th.dtype, shape=features.shape)


quan = InputDropout(mask_size=40, force_mask=True, erosion=0,
                    mask_gen_method='np')
quan.dim = 2
mask_arr = quan.np_gen_mask(features)

# Erosion = 0
masked_x_tensor = quan._link(tensor)
feed_dict = {tensor: features, quan.keep_mask: mask_arr}
masked_x_arr = sess.run(masked_x_tensor, feed_dict)
drop_mask_0 = sess.run(quan.drop_mask, feed_dict)

# Erosion = 2
quan.erosion = 6
_ = quan._link(tensor)
feed_dict = {tensor: features, quan.keep_mask: mask_arr}
drop_mask_2 = sess.run(quan.drop_mask, feed_dict)

data_dict = {}
data_dict['noisy image: $x$'] = features[0]
data_dict['$input: x\odot m$'] = masked_x_arr[0]
data_dict['$reference: x\odot (1-m)$'] = (features * drop_mask_0)[0]
data_dict['$input: x\odot  m$'] = masked_x_arr[0]
data_dict[r'$reference: x\odot\rm{erode}(1-m)$'] = (features * drop_mask_2)[0]
data_dict['$x\odot (1-m)$'] = (features * drop_mask_0)[0]
data_dict[r'$x\odot \rm{erode}(1-m)$'] = (features * drop_mask_2)[0]

Omma.visualize(data_dict, share_roi=True)





