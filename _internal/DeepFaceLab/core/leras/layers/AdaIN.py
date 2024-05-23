from core.leras import nn
tf = nn.tf

class AdaIN(nn.LayerBase):
    """
    """
    def __init__(self, in_ch, mlp_ch, kernel_initializer=None, dtype=None, **kwargs):
        self.in_ch = in_ch
        self.mlp_ch = mlp_ch
        self.kernel_initializer = kernel_initializer

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        kernel_initializer = self.kernel_initializer
        if kernel_initializer is None:
            kernel_initializer = tf.initializers.he_normal()

        self.weight1   = tf.get_variable("weight1", (self.mlp_ch, self.in_ch), dtype=self.dtype, initializer=kernel_initializer)
        self.bias1     = tf.get_variable("bias1",   (self.in_ch,), dtype=self.dtype, initializer=tf.initializers.zeros())
        self.weight2   = tf.get_variable("weight2", (self.mlp_ch, self.in_ch), dtype=self.dtype, initializer=kernel_initializer)
        self.bias2     = tf.get_variable("bias2",   (self.in_ch,), dtype=self.dtype, initializer=tf.initializers.zeros())

    def get_weights(self):
        return [self.weight1, self.bias1, self.weight2, self.bias2]

    def forward(self, inputs):
        x, mlp = inputs

        gamma = tf.matmul(mlp, self.weight1)
        gamma = tf.add(gamma, tf.reshape(self.bias1, (1,self.in_ch) ) )

        beta = tf.matmul(mlp, self.weight2)
        beta = tf.add(beta, tf.reshape(self.bias2, (1,self.in_ch) ) )


        if nn.data_format == "NHWC":
            shape = (-1,1,1,self.in_ch)
        else:
            shape = (-1,self.in_ch,1,1)

        x_mean = tf.reduce_mean(x, axis=nn.conv2d_spatial_axes, keepdims=True )
        x_std  = tf.math.reduce_std(x, axis=nn.conv2d_spatial_axes, keepdims=True ) + 1e-5

        x = (x - x_mean) / x_std
        x *= tf.reshape(gamma, shape)

        x += tf.reshape(beta, shape)

        return x

nn.AdaIN = AdaIN