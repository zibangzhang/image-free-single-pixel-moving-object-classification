import tensorflow as tf
import numpy as np
import scipy.misc
from scipy import io

reader = tf.train.NewCheckpointReader('./modelParameters/dcnet_mnist1.ckpt')
all_variables = reader.get_variable_to_shape_map()
w0 = reader.get_tensor("encode/weights1")
np.save('./pattern/weights1.npy',w0)
for i in range(w0.shape[3]):
    pattern = np.reshape(w0[:,:,:,i],[28,28])
    scipy.misc.imsave("./pattern/pattern%d.bmp"%(i+1), pattern)

weights1 = np.load('./pattern/weights1.npy')
io.savemat('./pattern/weights1.mat', {'weights1': weights1})