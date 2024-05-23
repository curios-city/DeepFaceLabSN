from core.leras import nn
tf = nn.tf

class DeepFakeArchi(nn.ArchiBase):
    """
    resolution

    mod     None - default
            'quick'

    opts    ''
            ''
            't'
    """
    def __init__(self, resolution, use_fp16=False, mod=None, opts=None):
        super().__init__()

        if opts is None:
            opts = ''


        conv_dtype = tf.float16 if use_fp16 else tf.float32
        
        if 'c' in opts:
            def act(x, alpha=0.1):
                return x*tf.cos(x)
        else:
            def act(x, alpha=0.1):
                return tf.nn.leaky_relu(x, alpha)
                
        if mod is None:
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch, kernel_size=self.kernel_size, strides=2, padding='SAME', dtype=conv_dtype)

                def forward(self, x):
                    x = self.conv1(x)
                    x = act(x, 0.1)
                    return x

                def get_out_ch(self):
                    return self.out_ch

            class DownscaleBlock(nn.ModelBase):
                def on_build(self, in_ch, ch, n_downscales, kernel_size):
                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size))
                        last_ch = self.downs[-1].get_out_ch()

                def forward(self, inp):
                    x = inp
                    for down in self.downs:
                        x = down(x)
                    return x

            class Upscale(nn.ModelBase):
                def on_build(self, in_ch, out_ch, kernel_size=3):
                    self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

                def forward(self, x):
                    x = self.conv1(x)
                    x = act(x, 0.1)
                    x = nn.depth_to_space(x, 2)
                    return x

            class ResidualBlock(nn.ModelBase):
                def on_build(self, ch, kernel_size=3):
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = act(x, 0.2)
                    x = self.conv2(x)
                    x = act(inp + x, 0.2)
                    return x

            class Encoder(nn.ModelBase):
                def __init__(self, in_ch, e_ch, **kwargs ):
                    self.in_ch = in_ch
                    self.e_ch = e_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    if 't' in opts:
                        self.down1 = Downscale(self.in_ch, self.e_ch, kernel_size=5)
                        self.res1 = ResidualBlock(self.e_ch)
                        self.down2 = Downscale(self.e_ch, self.e_ch*2, kernel_size=5)
                        self.down3 = Downscale(self.e_ch*2, self.e_ch*4, kernel_size=5)
                        self.down4 = Downscale(self.e_ch*4, self.e_ch*8, kernel_size=5)
                        self.down5 = Downscale(self.e_ch*8, self.e_ch*8, kernel_size=5)
                        self.res5 = ResidualBlock(self.e_ch*8)
                    else:
                        self.down1 = DownscaleBlock(self.in_ch, self.e_ch, n_downscales=4 if 't' not in opts else 5, kernel_size=5)

                def forward(self, x):
                    if use_fp16:
                        x = tf.cast(x, tf.float16)

                    if 't' in opts:
                        x = self.down1(x)
                        x = self.res1(x)
                        x = self.down2(x)
                        x = self.down3(x)
                        x = self.down4(x)
                        x = self.down5(x)
                        x = self.res5(x)
                    else:
                        x = self.down1(x)
                    x = nn.flatten(x)
                    if 'u' in opts:
                        x = nn.pixel_norm(x, axes=-1)

                    if use_fp16:
                        x = tf.cast(x, tf.float32)
                    return x

                def get_out_res(self, res):
                    return res // ( (2**4) if 't' not in opts else (2**5) )

                def get_out_ch(self):
                    return self.e_ch * 8

            lowest_dense_res = resolution // (32 if 'd' in opts else 16)

            class Inter(nn.ModelBase):
                def __init__(self, in_ch, ae_ch, ae_out_ch, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch

                    self.dense1 = nn.Dense( in_ch, ae_ch )
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                    if 't' not in opts:
                        self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

                def forward(self, inp):
                    x = inp
                    x = self.dense1(x)
                    x = self.dense2(x)
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)

                    if use_fp16:
                        x = tf.cast(x, tf.float16)

                    if 't' not in opts:
                        x = self.upscale1(x)

                    return x

                def get_out_res(self):
                    return lowest_dense_res * 2 if 't' not in opts else lowest_dense_res

                def get_out_ch(self):
                    return self.ae_out_ch

            class Decoder(nn.ModelBase):
                def on_build(self, in_ch, d_ch, d_mask_ch):
                    if 't' not in opts:
                        self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                        self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                        self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
                        self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                        self.res2 = ResidualBlock(d_ch*2, kernel_size=3)

                        self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                        self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                        self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)

                        self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)

                        if 'd' in opts:
                            self.out_conv1 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv2 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv3 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        else:
                            self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                    else:
                        self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                        self.upscale1 = Upscale(d_ch*8, d_ch*8, kernel_size=3)
                        self.upscale2 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                        self.upscale3 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
                        self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res1 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res2 = ResidualBlock(d_ch*4, kernel_size=3)
                        self.res3 = ResidualBlock(d_ch*2, kernel_size=3)

                        self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                        self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*8, kernel_size=3)
                        self.upscalem2 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                        self.upscalem3 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                        self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)

                        if 'd' in opts:
                            self.out_conv1 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv2 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.out_conv3 = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                            self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                            self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)
                        else:
                            self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)

                
                    
                def forward(self, z):
                    x = self.upscale0(z)
                    x = self.res0(x)
                    x = self.upscale1(x)
                    x = self.res1(x)
                    x = self.upscale2(x)
                    x = self.res2(x)

                    if 't' in opts:
                        x = self.upscale3(x)
                        x = self.res3(x)

                    if 'd' in opts:
                        x = tf.nn.sigmoid( nn.depth_to_space(tf.concat( (self.out_conv(x),
                                                                         self.out_conv1(x),
                                                                         self.out_conv2(x),
                                                                         self.out_conv3(x)), nn.conv2d_ch_axis), 2) )
                    else:
                        x = tf.nn.sigmoid(self.out_conv(x))


                    m = self.upscalem0(z)
                    m = self.upscalem1(m)
                    m = self.upscalem2(m)

                    if 't' in opts:
                        m = self.upscalem3(m)
                        if 'd' in opts:
                            m = self.upscalem4(m)
                    else:
                        if 'd' in opts:
                            m = self.upscalem3(m)

                    m = tf.nn.sigmoid(self.out_convm(m))

                    if use_fp16:
                        x = tf.cast(x, tf.float32)
                        m = tf.cast(m, tf.float32)

                    return x, m

        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder

nn.DeepFakeArchi = DeepFakeArchi