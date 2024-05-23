# -*- coding: utf-8 -*-

from core.leras import nn
tf = nn.tf

class XSeg(nn.ModelBase):
    
    def on_build (self, in_ch, base_ch, out_ch, resolution):
        if resolution is None:
            resolution = 256
        self.scale_ch= resolution/256
        
        # 定义一个卷积块
        class ConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):              
                # 定义卷积层，输入通道数为in_ch，输出通道数为out_ch，卷积核大小为3x3，padding方式为SAME
                self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=3, padding='SAME')
                # 定义FRN（Filter Response Normalization）归一化层，对卷积输出进行归一化
                self.frn = nn.FRNorm2D(out_ch)
                # 定义TLU（Thresholded Linear Unit）激活函数层，对归一化后的输出进行激活
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):                
                # 执行卷积操作
                x = self.conv(x)
                # 对卷积输出进行FRN归一化
                x = self.frn(x)
                # 对归一化后的输出进行TLU激活
                x = self.tlu(x)
                return x

        # 定义一个反卷积块
        class UpConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):
                # 定义反卷积层，输入通道数为in_ch，输出通道数为out_ch，卷积核大小为3x3，padding方式为SAME
                self.conv = nn.Conv2DTranspose(in_ch, out_ch, kernel_size=3, padding='SAME')
                # 定义FRN归一化层，对反卷积输出进行归一化
                self.frn = nn.FRNorm2D(out_ch)
                # 定义TLU激活函数层，对归一化后的反卷积输出进行激活
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):
                # 执行反卷积操作
                x = self.conv(x)
                # 对反卷积输出进行FRN归一化
                x = self.frn(x)
                # 对归一化后的反卷积输出进行TLU激活
                x = self.tlu(x)
                return x
                
        self.base_ch = base_ch

        self.conv01 = ConvBlock(in_ch, base_ch)
        self.conv02 = ConvBlock(base_ch, base_ch)
        self.bp0 = nn.BlurPool (filt_size=4)

        self.conv11 = ConvBlock(base_ch, base_ch*2)
        self.conv12 = ConvBlock(base_ch*2, base_ch*2)
        self.bp1 = nn.BlurPool (filt_size=3)

        self.conv21 = ConvBlock(base_ch*2, base_ch*4)
        self.conv22 = ConvBlock(base_ch*4, base_ch*4)
        self.bp2 = nn.BlurPool (filt_size=2)

        self.conv31 = ConvBlock(base_ch*4, base_ch*8)
        self.conv32 = ConvBlock(base_ch*8, base_ch*8)
        self.conv33 = ConvBlock(base_ch*8, base_ch*8)
        self.bp3 = nn.BlurPool (filt_size=2)

        self.conv41 = ConvBlock(base_ch*8, base_ch*8)
        self.conv42 = ConvBlock(base_ch*8, base_ch*8)
        self.conv43 = ConvBlock(base_ch*8, base_ch*8)
        self.bp4 = nn.BlurPool (filt_size=2)
        
        self.conv51 = ConvBlock(base_ch*8, base_ch*8)
        self.conv52 = ConvBlock(base_ch*8, base_ch*8)
        self.conv53 = ConvBlock(base_ch*8, base_ch*8)
        self.bp5 = nn.BlurPool (filt_size=2)
        
        self.dense1 = nn.Dense (int(4096*self.scale_ch**2), 512)
        self.dense2 = nn.Dense ( 512,int(4096*self.scale_ch**2))
    
        self.up5 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv53 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv52 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv51 = ConvBlock(base_ch*8, base_ch*8)
        
        self.up4 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv43 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv42 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv41 = ConvBlock(base_ch*8, base_ch*8)

        self.up3 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv33 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv32 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv31 = ConvBlock(base_ch*8, base_ch*8)

        self.up2 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv22 = ConvBlock(base_ch*8, base_ch*4)
        self.uconv21 = ConvBlock(base_ch*4, base_ch*4)

        self.up1 = UpConvBlock (base_ch*4, base_ch*2)
        self.uconv12 = ConvBlock(base_ch*4, base_ch*2)
        self.uconv11 = ConvBlock(base_ch*2, base_ch*2)

        self.up0 = UpConvBlock (base_ch*2, base_ch)
        self.uconv02 = ConvBlock(base_ch*2, base_ch)
        self.uconv01 = ConvBlock(base_ch, base_ch)
        self.out_conv = nn.Conv2D (base_ch, out_ch, kernel_size=3, padding='SAME')
    
        
    def forward(self, inp, pretrain=False):
        x = inp
        x = self.conv01(x)
        x = x0 = self.conv02(x)
        x = self.bp0(x)
        x = self.conv11(x)
        x = x1 = self.conv12(x)
        x = self.bp1(x)
        x = self.conv21(x)
        x = x2 = self.conv22(x)
        x = self.bp2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x3 = self.conv33(x)
        x = self.bp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = x4 = self.conv43(x)
        x = self.bp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = x5 = self.conv53(x)
        x = self.bp5(x)
        
        x = nn.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = nn.reshape_4D (x, int(4*self.scale_ch), int(4*self.scale_ch), self.base_ch*8 )
                          
        x = self.up5(x)
        if pretrain:
            x5 = tf.zeros_like(x5)
        x = self.uconv53(tf.concat([x,x5],axis=nn.conv2d_ch_axis))
        x = self.uconv52(x)
        x = self.uconv51(x)
        
        x = self.up4(x)
        if pretrain:
            x4 = tf.zeros_like(x4)
        x = self.uconv43(tf.concat([x,x4],axis=nn.conv2d_ch_axis))
        x = self.uconv42(x)
        x = self.uconv41(x)

        x = self.up3(x)
        if pretrain:
            x3 = tf.zeros_like(x3)
        x = self.uconv33(tf.concat([x,x3],axis=nn.conv2d_ch_axis))
        x = self.uconv32(x)
        x = self.uconv31(x)

        x = self.up2(x)
        if pretrain:
            x2 = tf.zeros_like(x2)
        x = self.uconv22(tf.concat([x,x2],axis=nn.conv2d_ch_axis))
        x = self.uconv21(x)

        x = self.up1(x)
        if pretrain:
            x1 = tf.zeros_like(x1)
        x = self.uconv12(tf.concat([x,x1],axis=nn.conv2d_ch_axis))
        x = self.uconv11(x)

        x = self.up0(x)
        if pretrain:
            x0 = tf.zeros_like(x0)
        x = self.uconv02(tf.concat([x,x0],axis=nn.conv2d_ch_axis))
        x = self.uconv01(x)

        logits = self.out_conv(x)
        return logits, tf.nn.sigmoid(logits)

nn.XSeg = XSeg