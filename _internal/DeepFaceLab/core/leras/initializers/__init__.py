import numpy as np
from tensorflow.python.ops import init_ops

from core.leras import nn

tf = nn.tf

from .CA import CAInitializerSubprocessor

class initializers():
    class ca (init_ops.Initializer):
        def __call__(self, shape, dtype=None, partition_info=None):
            return tf.zeros( shape, dtype=dtype, name="_cai_")

        @staticmethod
        def generate_batch( data_list, eps_std=0.05 ):
            # list of (shape, np.dtype)
            return CAInitializerSubprocessor (data_list).run()

nn.initializers = initializers
