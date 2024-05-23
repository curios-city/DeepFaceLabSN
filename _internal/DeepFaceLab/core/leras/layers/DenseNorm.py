from core.leras import nn
tf = nn.tf

class DenseNorm(nn.LayerBase):
    def __init__(self, dense=False, eps=1e-06, dtype=None, **kwargs):
        self.dense = dense        
        if dtype is None:
            dtype = nn.floatx
        self.eps = tf.constant(eps, dtype=dtype, name="epsilon")

        super().__init__(**kwargs)

    def __call__(self, x):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        
nn.DenseNorm = DenseNorm