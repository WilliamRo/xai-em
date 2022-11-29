import numpy as np
# Put this line before importing tensorflow to get rid of future warnings
from tframe import console
import tensorflow as tf


console.suppress_logging()
console.start('Random Graph')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
random_tensor = tf.random.uniform([2, 2])
tensor_a = random_tensor * 10
tensor_b = random_tensor * 100

sess = tf.get_default_session()

# Print results
for i in range(10000):
  results = sess.run([tensor_a, tensor_b])
  delta = np.max(np.abs(results[0] - results[1] / 10))
  if delta > 0.1:
    print(i+1)
    print(delta)
# =============================================================================
# End of the script
# =============================================================================
console.end()
