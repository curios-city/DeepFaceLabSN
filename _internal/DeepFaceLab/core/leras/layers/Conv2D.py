import numpy as np
from core.leras import nn
tf = nn.tf

class Conv2D(nn.LayerBase):
    """
    default kernel_initializer - CA
    use_wscale  bool enables equalized learning rate, if kernel_initializer is None, it will be forced to random_normal


    """
    def __init__(self, in_ch, out_ch, kernel_size, strides=1, padding='SAME', dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        if not isinstance(strides, int):
            raise ValueError ("strides must be an int type")
        if not isinstance(dilations, int):
            raise ValueError ("dilations must be an int type")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx

        if isinstance(padding, str):
            if padding == "SAME":
                padding = ( (kernel_size - 1) * dilations + 1 ) // 2
            elif padding == "VALID":
                padding = None
            else:
                raise ValueError ("Wrong padding type. Should be VALID SAME or INT or 4x INTs")
        else:
            padding = int(padding)
            
        

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
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
            he_std = gain / np.sqrt(fan_in)
            self.wscale = tf.constant(he_std, dtype=self.dtype )
            if kernel_initializer is None:
                kernel_initializer = tf.initializers.random_normal(0, 1.0, dtype=self.dtype)

        #if kernel_initializer is None:
        #    kernel_initializer = nn.initializers.ca()

        self.weight = tf.get_variable("weight", (self.kernel_size,self.kernel_size,self.in_ch,self.out_ch), dtype=self.dtype, initializer=kernel_initializer, trainable=self.trainable )

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

        padding = self.padding
        if padding is not None:
            if nn.data_format == "NHWC":
                padding = [ [0,0], [padding,padding], [padding,padding], [0,0] ]
            else:
                padding = [ [0,0], [0,0], [padding,padding], [padding,padding] ]
            x = tf.pad (x, padding, mode='CONSTANT')
        
        strides = self.strides
        if nn.data_format == "NHWC":
            strides = [1,strides,strides,1]
        else:
            strides = [1,1,strides,strides]

        dilations = self.dilations
        if nn.data_format == "NHWC":
            dilations = [1,dilations,dilations,1]
        else:
            dilations = [1,1,dilations,dilations]
            
        x = tf.nn.conv2d(x, weight, strides, 'VALID', dilations=dilations, data_format=nn.data_format)
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
nn.Conv2D = Conv2D