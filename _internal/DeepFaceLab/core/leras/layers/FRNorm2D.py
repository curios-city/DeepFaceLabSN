from core.leras import nn
tf = nn.tf

class FRNorm2D(nn.LayerBase):
    """
    Tensorflow implementation of
    Filter Response Normalization Layer: Eliminating Batch Dependence in theTraining of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        self.weight      = tf.get_variable("weight", (self.in_ch,), dtype=self.dtype, initializer=tf.initializers.ones() )
        self.bias        = tf.get_variable("bias",   (self.in_ch,), dtype=self.dtype, initializer=tf.initializers.zeros() )
        self.eps         = tf.get_variable("eps",    (1,), dtype=self.dtype, initializer=tf.initializers.constant(1e-6) )

    def get_weights(self):
        return [self.weight, self.bias, self.eps]

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.in_ch)
        else:
            shape = (1,self.in_ch,1,1)
        weight       = tf.reshape ( self.weight, shape )
        bias         = tf.reshape ( self.bias  , shape )
        nu2 = tf.reduce_mean(tf.square(x), axis=nn.conv2d_spatial_axes, keepdims=True)
        x = x * ( 1.0/tf.sqrt(nu2 + tf.abs(self.eps) ) )

        return x*weight + bias
nn.FRNorm2D = FRNorm2D