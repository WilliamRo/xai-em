from bd_core import th
from tframe import context, console, Predictor

import bd_du as du
import t1_quan as module



console.suppress_logging()

th.developer_code = 'block_activate'
module.main(None)

th.train = 0
th.overwrite = 1
th.force_mask = 1
th.mask_size = 16

model: Predictor = th.model()

fetcher = context.get_collection_by_key('quan')['masked_input']
data_set = du.load_data().data_for_validation.as_parent_data_set
masked_input = model.evaluate(fetcher, data_set)


from xem.ui.omma import Omma

Omma.visualize({'input': data_set.features[0], 'masked_input': masked_input[0]},
               title=True, mini_map=True, share_roi=True, init_depth=100)





