import numpy as np
from core.leras import nn
tf = nn.tf

class Conv2DTranspose(nn.LayerBase):
    """
    use_wscale      enables weight scale (equalized learning rate)
                    if kernel_initializer is None, it will be forced to random_normal
    """
    def __init__(self, in_ch, out_ch, kernel_size, strides=2, padding='SAME', use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        if not isinstance(strides, int):
            raise ValueError ("strides must be an int type")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        kernel_initializer = self.kernel_initializer
        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2)
            fan_in = self.kernel_size*self.kernel_size*self.in_ch
            he_std = gain / np.sqrt(fan_in) # He init
            self.wscale = tf.constant(he_std, dtype=self.dtype )
            if kernel_initializer is None:
                kernel_initializer = tf.initializers.random_normal(0, 1.0, dtype=self.dtype)

        #if kernel_initializer is None:
        #    kernel_initializer = nn.initializers.ca()
        self.weight = tf.get_variable("weight", (self.kernel_size,self.kernel_size,self.out_ch,self.in_ch), dtype=self.dtype, initializer=kernel_initializer, trainable=self.trainable )

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
        shape = x.shape

        if nn.data_format == "NHWC":
            h,w,c = shape[1], shape[2], shape[3]
            output_shape = tf.stack ( (tf.shape(x)[0],
                                    self.deconv_length(w, self.strides, self.kernel_size, self.padding),
                                    self.deconv_length(h, self.strides, self.kernel_size, self.padding),
                                    self.out_ch) )

            strides = [1,self.strides,self.strides,1]
        else:
            c,h,w = shape[1], shape[2], shape[3]
            output_shape = tf.stack ( (tf.shape(x)[0],
                                        self.out_ch,
                                        self.deconv_length(w, self.strides, self.kernel_size, self.padding),
                                        self.deconv_length(h, self.strides, self.kernel_size, self.padding),
                                        ) )
            strides = [1,1,self.strides,self.strides]
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        x = tf.nn.conv2d_transpose(x, weight, output_shape, strides, padding=self.padding, data_format=nn.data_format)

        if self.use_bias:
            if nn.data_format == "NHWC":
                bias = tf.reshape (self.bias, (1,1,1,self.out_ch) )
            else:
                bias = tf.reshape (self.bias, (1,self.out_ch,1,1) )
            x = tf.add(x, bias)
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "

        return r

    def deconv_length(self, dim_size, stride_size, kernel_size, padding):
        assert padding in {'SAME', 'VALID', 'FULL'}
        if dim_size is None:
            return None
        if padding == 'VALID':
            dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
        elif padding == 'FULL':
            dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
        elif padding == 'SAME':
            dim_size = dim_size * stride_size
        return dim_size
nn.Conv2DTranspose = Conv2DTranspose