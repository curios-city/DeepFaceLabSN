import numpy as np
from core.leras import nn
tf = nn.tf

class Dense(nn.LayerBase):
    def __init__(self, in_ch, out_ch, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        """
        use_wscale          enables weight scale (equalized learning rate)
                            if kernel_initializer is None, it will be forced to random_normal

        maxout_ch     https://link.springer.com/article/10.1186/s40537-019-0233-0
                            typical 2-4 if you want to enable DenseMaxout behaviour
        """
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.maxout_ch = maxout_ch
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        if dtype is None:
            dtype = nn.floatx

        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        if self.maxout_ch > 1:
            weight_shape = (self.in_ch,self.out_ch*self.maxout_ch)
        else:
            weight_shape = (self.in_ch,self.out_ch)

        kernel_initializer = self.kernel_initializer

        if self.use_wscale:
            gain = 1.0
            fan_in = np.prod( weight_shape[:-1] )
            he_std = gain / np.sqrt(fan_in) # He init
            self.wscale = tf.constant(he_std, dtype=self.dtype )
            if kernel_initializer is None:
                kernel_initializer = tf.initializers.random_normal(0, 1.0, dtype=self.dtype)

        if kernel_initializer is None:
            kernel_initializer = tf.initializers.glorot_uniform(dtype=self.dtype)

        self.weight = tf.get_variable("weight", weight_shape, dtype=self.dtype, initializer=kernel_initializer, trainable=self.trainable )

        if self.use_bias:
            bias_initializer = self.bias_initializer
            if bias_initializer is None:
                bias_initializer = tf.initializers.zeros(dtype=self.dtype)
            self.bias = tf.get_variable("bias", (self.out_ch,), dtype=self.dtype, initializer=bias_initializer, trainable=self.trainable )

    def get_weights(self):
        weights = [self.weight]
        if self.use_bias:
            weights += [self.bias]
        return weights

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        x = tf.matmul(x, weight)

        if self.maxout_ch > 1:
            x = tf.reshape (x, (-1, self.out_ch, self.maxout_ch) )
            x = tf.reduce_max(x, axis=-1)

        if self.use_bias:
            x = tf.add(x, tf.reshape(self.bias, (1,self.out_ch) ) )

        return x
nn.Dense = Dense