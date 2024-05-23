import multiprocessing
import operator
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *
from core.cv2ex import *

class AMPModel(ModelBase):

    #override
    def on_initialize_options(self):
        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 224)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'wf')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_inter_dims         = self.options['inter_dims']         = self.load_or_def_option('inter_dims', 1024)

        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        default_morph_factor       = self.options['morph_factor']       = self.options.get('morph_factor', 0.5)
        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)
        default_blur_out_mask      = self.options['blur_out_mask']      = self.load_or_def_option('blur_out_mask', False)
        default_lr_dropout         = self.options['lr_dropout']         = self.load_or_def_option('lr_dropout', 'n')
        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(8)

        if self.is_first_run():
            resolution = io.input_int("Resolution", default_resolution, add_info="64-640", help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 32 .")
            resolution = np.clip ( (resolution // 32) * 32, 64, 640)
            self.options['resolution'] = resolution
            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['f','wf','head'], help_message="whole face / head").lower()


        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', 64)

        default_d_mask_dims        = default_d_dims // 3
        default_d_mask_dims        += default_d_mask_dims % 2
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            self.options['ae_dims']    = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )
            self.options['inter_dims'] = np.clip ( io.input_int("Inter dimensions", default_inter_dims, add_info="32-2048", help_message="Should be equal or more than AutoEncoder dimensions. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 2048 )

            e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

            morph_factor = np.clip ( io.input_number ("Morph factor.", default_morph_factor, add_info="0.1 .. 0.5", help_message="Typical fine value is 0.5"), 0.1, 0.5 )
            self.options['morph_factor'] = morph_factor

        if self.is_first_run() or ask_override:
            self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution of samples", default_uniform_yaw, help_message='Helps to fix blurry side faces due to small amount of them in the faceset.')
            self.options['blur_out_mask'] = io.input_bool ("Blur out mask", default_blur_out_mask, help_message='Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required.')
            self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. \nn - disabled.\ny - enabled\ncpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration.")

        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size     = self.options['gan_patch_size']     = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
        default_gan_dims           = self.options['gan_dims']           = self.load_or_def_option('gan_dims', 16)

        if self.is_first_run() or ask_override:
            self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

            self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 5.0", help_message="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1"), 0.0, 5.0 )

            if self.options['gan_power'] != 0.0:
                gan_patch_size = np.clip ( io.input_int("GAN patch size", default_gan_patch_size, add_info="3-640", help_message="The higher patch size, the higher the quality, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is resolution / 8." ), 3, 640 )
                self.options['gan_patch_size'] = gan_patch_size

                gan_dims = np.clip ( io.input_int("GAN dimensions", default_gan_dims, add_info="4-512", help_message="The dimensions of the GAN network. The higher dimensions, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is 16." ), 4, 512 )
                self.options['gan_dims'] = gan_dims

            self.options['ct_mode'] = io.input_str (f"Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="Change color distribution of src samples close to dst samples. If src faceset is deverse enough, then lct mode is fine in most cases.")
            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

        self.gan_model_changed = (default_gan_patch_size != self.options['gan_patch_size']) or (default_gan_dims != self.options['gan_dims'])

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        input_ch=3
        resolution  = self.resolution = self.options['resolution']
        e_dims      = self.options['e_dims']
        ae_dims     = self.options['ae_dims']
        inter_dims  = self.inter_dims = self.options['inter_dims']
        inter_res   = self.inter_res = resolution // 32
        d_dims      = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        face_type   = self.face_type = {'f'    : FaceType.FULL,
                                        'wf'   : FaceType.WHOLE_FACE,
                                        'head' : FaceType.HEAD}[ self.options['face_type'] ]
        morph_factor = self.options['morph_factor']
        gan_power    = self.gan_power = self.options['gan_power']
        random_warp  = self.options['random_warp']

        blur_out_mask = self.options['blur_out_mask']

        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        use_fp16 = False
        if self.is_exporting:
            use_fp16 = io.input_bool ("Export quantized?", False, help_message='Makes the exported model faster. If you have problems, disable this option.')

        conv_dtype = tf.float16 if use_fp16 else tf.float32

        class Downscale(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=5 ):
                self.conv1 = nn.Conv2D( in_ch, out_ch, kernel_size=kernel_size, strides=2, padding='SAME', dtype=conv_dtype)

            def forward(self, x):
                return tf.nn.leaky_relu(self.conv1(x), 0.1)

        class Upscale(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D(in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

            def forward(self, x):
                x = nn.depth_to_space(tf.nn.leaky_relu(self.conv1(x), 0.1), 2)
                return x

        class ResidualBlock(nn.ModelBase):
            def on_build(self, ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

            def forward(self, inp):
                x = self.conv1(inp)
                x = tf.nn.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = tf.nn.leaky_relu(inp+x, 0.2)
                return x

        class Encoder(nn.ModelBase):
            def on_build(self):
                self.down1 = Downscale(input_ch, e_dims, kernel_size=5)
                self.res1 = ResidualBlock(e_dims)
                self.down2 = Downscale(e_dims, e_dims*2, kernel_size=5)
                self.down3 = Downscale(e_dims*2, e_dims*4, kernel_size=5)
                self.down4 = Downscale(e_dims*4, e_dims*8, kernel_size=5)
                self.down5 = Downscale(e_dims*8, e_dims*8, kernel_size=5)
                self.res5 = ResidualBlock(e_dims*8)
                self.dense1 = nn.Dense( (( resolution//(2**5) )**2) * e_dims*8, ae_dims )

            def forward(self, x):
                if use_fp16:
                    x = tf.cast(x, tf.float16)
                x = self.down1(x)
                x = self.res1(x)
                x = self.down2(x)
                x = self.down3(x)
                x = self.down4(x)
                x = self.down5(x)
                x = self.res5(x)
                if use_fp16:
                    x = tf.cast(x, tf.float32)
                x = nn.pixel_norm(nn.flatten(x), axes=-1)
                x = self.dense1(x)
                return x


        class Inter(nn.ModelBase):
            def on_build(self):
                self.dense2 = nn.Dense(ae_dims, inter_res * inter_res * inter_dims)

            def forward(self, inp):
                x = inp
                x = self.dense2(x)
                x = nn.reshape_4D (x, inter_res, inter_res, inter_dims)
                return x


        class Decoder(nn.ModelBase):
            def on_build(self ):
                self.upscale0 = Upscale(inter_dims, d_dims*8, kernel_size=3)
                self.upscale1 = Upscale(d_dims*8, d_dims*8, kernel_size=3)
                self.upscale2 = Upscale(d_dims*8, d_dims*4, kernel_size=3)
                self.upscale3 = Upscale(d_dims*4, d_dims*2, kernel_size=3)

                self.res0 = ResidualBlock(d_dims*8, kernel_size=3)
                self.res1 = ResidualBlock(d_dims*8, kernel_size=3)
                self.res2 = ResidualBlock(d_dims*4, kernel_size=3)
                self.res3 = ResidualBlock(d_dims*2, kernel_size=3)

                self.upscalem0 = Upscale(inter_dims, d_mask_dims*8, kernel_size=3)
                self.upscalem1 = Upscale(d_mask_dims*8, d_mask_dims*8, kernel_size=3)
                self.upscalem2 = Upscale(d_mask_dims*8, d_mask_dims*4, kernel_size=3)
                self.upscalem3 = Upscale(d_mask_dims*4, d_mask_dims*2, kernel_size=3)
                self.upscalem4 = Upscale(d_mask_dims*2, d_mask_dims*1, kernel_size=3)
                self.out_convm = nn.Conv2D( d_mask_dims*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)

                self.out_conv  = nn.Conv2D( d_dims*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)
                self.out_conv1 = nn.Conv2D( d_dims*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                self.out_conv2 = nn.Conv2D( d_dims*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
                self.out_conv3 = nn.Conv2D( d_dims*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)

            def forward(self, z):
                if use_fp16:
                    z = tf.cast(z, tf.float16)

                x = self.upscale0(z)
                x = self.res0(x)
                x = self.upscale1(x)
                x = self.res1(x)
                x = self.upscale2(x)
                x = self.res2(x)
                x = self.upscale3(x)
                x = self.res3(x)

                x = tf.nn.sigmoid( nn.depth_to_space(tf.concat( (self.out_conv(x),
                                                                 self.out_conv1(x),
                                                                 self.out_conv2(x),
                                                                 self.out_conv3(x)), nn.conv2d_ch_axis), 2) )
                m = self.upscalem0(z)
                m = self.upscalem1(m)
                m = self.upscalem2(m)
                m = self.upscalem3(m)
                m = self.upscalem4(m)
                m = tf.nn.sigmoid(self.out_convm(m))

                if use_fp16:
                    x = tf.cast(x, tf.float32)
                    m = tf.cast(m, tf.float32)
                return x, m

        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = nn.tf_default_device_name if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        bgr_shape = self.bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)
        self.model_filename_list = []

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape, name='warped_src')
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape, name='warped_dst')

            self.target_src = tf.placeholder (nn.floatx, bgr_shape, name='target_src')
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape, name='target_dst')

            self.target_srcm    = tf.placeholder (nn.floatx, mask_shape, name='target_srcm')
            self.target_srcm_em = tf.placeholder (nn.floatx, mask_shape, name='target_srcm_em')
            self.target_dstm    = tf.placeholder (nn.floatx, mask_shape, name='target_dstm')
            self.target_dstm_em = tf.placeholder (nn.floatx, mask_shape, name='target_dstm_em')

            self.morph_value_t = tf.placeholder (nn.floatx, (1,), name='morph_value_t')

        # Initializing model classes
        with tf.device (models_opt_device):
            self.encoder = Encoder(name='encoder')
            self.inter_src = Inter(name='inter_src')
            self.inter_dst = Inter(name='inter_dst')
            self.decoder = Decoder(name='decoder')

            self.model_filename_list += [   [self.encoder,  'encoder.npy'],
                                            [self.inter_src, 'inter_src.npy'],
                                            [self.inter_dst , 'inter_dst.npy'],
                                            [self.decoder , 'decoder.npy'] ]

            if self.is_training:
                # Initialize optimizers
                clipnorm = 1.0 if self.options['clipgrad'] else 0.0
                lr_dropout = 0.3 if self.options['lr_dropout'] in ['y','cpu'] else 1.0

                self.G_weights = self.encoder.get_weights() + self.decoder.get_weights()

                #if random_warp:
                #    self.G_weights += self.inter_src.get_weights() + self.inter_dst.get_weights()

                self.src_dst_opt = nn.AdaBelief(lr=5e-5, lr_dropout=lr_dropout, clipnorm=clipnorm, name='src_dst_opt')
                self.src_dst_opt.initialize_variables (self.G_weights, vars_on_cpu=optimizer_vars_on_cpu)
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

                if gan_power != 0:
                    self.GAN = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=input_ch, base_ch=self.options['gan_dims'], name="GAN")
                    self.GAN_opt = nn.AdaBelief(lr=5e-5, lr_dropout=lr_dropout, clipnorm=clipnorm, name='GAN_opt')
                    self.GAN_opt.initialize_variables ( self.GAN.get_weights(), vars_on_cpu=optimizer_vars_on_cpu)
                    self.model_filename_list += [ [self.GAN, 'GAN.npy'],
                                                  [self.GAN_opt, 'GAN_opt.npy'] ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            # Compute losses per GPU
            gpu_pred_src_src_list = []
            gpu_pred_dst_dst_list = []
            gpu_pred_src_dst_list = []
            gpu_pred_src_srcm_list = []
            gpu_pred_dst_dstm_list = []
            gpu_pred_src_dstm_list = []

            gpu_src_losses = []
            gpu_dst_losses = []
            gpu_G_loss_gradients = []
            gpu_GAN_loss_gradients = []

            def DLossOnes(logits):
                return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits), logits=logits), axis=[1,2,3])

            def DLossZeros(logits):
                return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits), logits=logits), axis=[1,2,3])

            for gpu_id in range(gpu_count):
                with tf.device( f'/{devices[gpu_id].tf_dev_type}:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                        gpu_warped_src      = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst      = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src      = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst      = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm     = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_srcm_em  = self.target_srcm_em[batch_slice,:,:,:]
                        gpu_target_dstm     = self.target_dstm[batch_slice,:,:,:]
                        gpu_target_dstm_em  = self.target_dstm_em[batch_slice,:,:,:]

                    # process model tensors
                    gpu_src_code = self.encoder (gpu_warped_src)
                    gpu_dst_code = self.encoder (gpu_warped_dst)

                    gpu_src_inter_src_code, gpu_src_inter_dst_code = self.inter_src (gpu_src_code), self.inter_dst (gpu_src_code)
                    gpu_dst_inter_src_code, gpu_dst_inter_dst_code = self.inter_src (gpu_dst_code), self.inter_dst (gpu_dst_code)

                    inter_dims_bin = int(inter_dims*morph_factor)
                    with tf.device(f'/CPU:0'):
                        inter_rnd_binomial = tf.stack([tf.random.shuffle(tf.concat([tf.tile(tf.constant([1], tf.float32), ( inter_dims_bin, )),
                                                                                    tf.tile(tf.constant([0], tf.float32), ( inter_dims-inter_dims_bin, ))], 0 )) for _ in range(bs_per_gpu)], 0)

                        inter_rnd_binomial = tf.stop_gradient(inter_rnd_binomial[...,None,None])

                    gpu_src_code = gpu_src_inter_src_code * inter_rnd_binomial + gpu_src_inter_dst_code * (1-inter_rnd_binomial)
                    gpu_dst_code = gpu_dst_inter_dst_code

                    inter_dims_slice = tf.cast(inter_dims*self.morph_value_t[0], tf.int32)
                    gpu_src_dst_code = tf.concat( (tf.slice(gpu_dst_inter_src_code, [0,0,0,0],   [-1, inter_dims_slice , inter_res, inter_res]),
                                                   tf.slice(gpu_dst_inter_dst_code, [0,inter_dims_slice,0,0], [-1,inter_dims-inter_dims_slice, inter_res,inter_res]) ), 1 )

                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)

                    gpu_pred_src_src_list.append(gpu_pred_src_src), gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst), gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst), gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    gpu_target_srcm_anti = 1-gpu_target_srcm
                    gpu_target_dstm_anti = 1-gpu_target_dstm

                    gpu_target_srcm_gblur = nn.gaussian_blur(gpu_target_srcm, resolution // 32)
                    gpu_target_dstm_gblur = nn.gaussian_blur(gpu_target_dstm, resolution // 32)

                    gpu_target_srcm_blur = tf.clip_by_value(gpu_target_srcm_gblur, 0, 0.5) * 2
                    gpu_target_dstm_blur = tf.clip_by_value(gpu_target_dstm_gblur, 0, 0.5) * 2
                    gpu_target_srcm_anti_blur = 1.0-gpu_target_srcm_blur
                    gpu_target_dstm_anti_blur = 1.0-gpu_target_dstm_blur

                    if blur_out_mask:
                        sigma = resolution / 128
                        
                        x = nn.gaussian_blur(gpu_target_src*gpu_target_srcm_anti, sigma)
                        y = 1-nn.gaussian_blur(gpu_target_srcm, sigma) 
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)                        
                        gpu_target_src = gpu_target_src*gpu_target_srcm + (x/y)*gpu_target_srcm_anti
                        
                        x = nn.gaussian_blur(gpu_target_dst*gpu_target_dstm_anti, sigma)
                        y = 1-nn.gaussian_blur(gpu_target_dstm, sigma) 
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)                        
                        gpu_target_dst = gpu_target_dst*gpu_target_dstm + (x/y)*gpu_target_dstm_anti

                    gpu_target_src_masked = gpu_target_src*gpu_target_srcm_blur
                    gpu_target_dst_masked = gpu_target_dst*gpu_target_dstm_blur
                    gpu_target_src_anti_masked = gpu_target_src*gpu_target_srcm_anti_blur
                    gpu_target_dst_anti_masked = gpu_target_dst*gpu_target_dstm_anti_blur

                    gpu_pred_src_src_masked = gpu_pred_src_src*gpu_target_srcm_blur
                    gpu_pred_dst_dst_masked = gpu_pred_dst_dst*gpu_target_dstm_blur
                    gpu_pred_src_src_anti_masked = gpu_pred_src_src*gpu_target_srcm_anti_blur
                    gpu_pred_dst_dst_anti_masked = gpu_pred_dst_dst*gpu_target_dstm_anti_blur

                    # Structural loss
                    gpu_src_loss =  tf.reduce_mean (5*nn.dssim(gpu_target_src_masked, gpu_pred_src_src_masked, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean (5*nn.dssim(gpu_target_src_masked, gpu_pred_src_src_masked, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                    gpu_dst_loss =  tf.reduce_mean (5*nn.dssim(gpu_target_dst_masked, gpu_pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean (5*nn.dssim(gpu_target_dst_masked, gpu_pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/23.2) ), axis=[1])

                    # Pixel loss
                    gpu_src_loss += tf.reduce_mean (10*tf.square(gpu_target_src_masked-gpu_pred_src_src_masked), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean (10*tf.square(gpu_target_dst_masked-gpu_pred_dst_dst_masked), axis=[1,2,3])

                    # Eyes+mouth prio loss
                    gpu_src_loss += tf.reduce_mean (300*tf.abs (gpu_target_src*gpu_target_srcm_em-gpu_pred_src_src*gpu_target_srcm_em), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean (300*tf.abs (gpu_target_dst*gpu_target_dstm_em-gpu_pred_dst_dst*gpu_target_dstm_em), axis=[1,2,3])

                    # Mask loss
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]
                    gpu_G_loss = gpu_src_loss + gpu_dst_loss
                    # dst-dst background weak loss
                    gpu_G_loss += tf.reduce_mean(0.1*tf.square(gpu_pred_dst_dst_anti_masked-gpu_target_dst_anti_masked),axis=[1,2,3] )
                    gpu_G_loss += 0.000001*nn.total_variation_mse(gpu_pred_dst_dst_anti_masked)


                    if gan_power != 0:
                        gpu_pred_src_src_d, gpu_pred_src_src_d2 = self.GAN(gpu_pred_src_src_masked)
                        gpu_pred_dst_dst_d, gpu_pred_dst_dst_d2 = self.GAN(gpu_pred_dst_dst_masked)
                        gpu_target_src_d, gpu_target_src_d2 = self.GAN(gpu_target_src_masked)
                        gpu_target_dst_d, gpu_target_dst_d2 = self.GAN(gpu_target_dst_masked)

                        gpu_GAN_loss = (DLossOnes (gpu_target_src_d)   + DLossOnes (gpu_target_src_d2) + \
                                        DLossZeros(gpu_pred_src_src_d) + DLossZeros(gpu_pred_src_src_d2) + \
                                        DLossOnes (gpu_target_dst_d)   + DLossOnes (gpu_target_dst_d2) + \
                                        DLossZeros(gpu_pred_dst_dst_d) + DLossZeros(gpu_pred_dst_dst_d2)
                                        ) * (1.0 / 8)

                        gpu_GAN_loss_gradients += [ nn.gradients (gpu_GAN_loss, self.GAN.get_weights() ) ]

                        gpu_G_loss += (DLossOnes(gpu_pred_src_src_d) + DLossOnes(gpu_pred_src_src_d2) + \
                                       DLossOnes(gpu_pred_dst_dst_d) + DLossOnes(gpu_pred_dst_dst_d2)
                                      ) * gan_power

                        # Minimal src-src-bg rec with total_variation_mse to suppress random bright dots from gan
                        gpu_G_loss += 0.000001*nn.total_variation_mse(gpu_pred_src_src)
                        gpu_G_loss += 0.02*tf.reduce_mean(tf.square(gpu_pred_src_src_anti_masked-gpu_target_src_anti_masked),axis=[1,2,3] )

                    gpu_G_loss_gradients += [ nn.gradients ( gpu_G_loss, self.G_weights ) ]

            # Average losses and gradients, and create optimizer update ops
            with tf.device(f'/CPU:0'):
                pred_src_src  = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst  = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst  = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

            with tf.device (models_opt_device):
                src_loss = tf.concat(gpu_src_losses, 0)
                dst_loss = tf.concat(gpu_dst_losses, 0)
                train_op = self.src_dst_opt.get_update_op (nn.average_gv_list (gpu_G_loss_gradients))

                if gan_power != 0:
                    GAN_train_op = self.GAN_opt.get_update_op (nn.average_gv_list(gpu_GAN_loss_gradients) )

            # Initializing training and view functions
            def train(warped_src, target_src, target_srcm, target_srcm_em,  \
                              warped_dst, target_dst, target_dstm, target_dstm_em, ):
                s, d, _ = nn.tf_sess.run ([src_loss, dst_loss, train_op],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm:target_srcm,
                                                       self.target_srcm_em:target_srcm_em,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm:target_dstm,
                                                       self.target_dstm_em:target_dstm_em,
                                                       })
                return s, d
            self.train = train

            if gan_power != 0:
                def GAN_train(warped_src, target_src, target_srcm, target_srcm_em,  \
                              warped_dst, target_dst, target_dstm, target_dstm_em, ):
                    nn.tf_sess.run ([GAN_train_op], feed_dict={self.warped_src :warped_src,
                                                               self.target_src :target_src,
                                                               self.target_srcm:target_srcm,
                                                               self.target_srcm_em:target_srcm_em,
                                                               self.warped_dst :warped_dst,
                                                               self.target_dst :target_dst,
                                                               self.target_dstm:target_dstm,
                                                               self.target_dstm_em:target_dstm_em})
                self.GAN_train = GAN_train

            def AE_view(warped_src, warped_dst, morph_value):
                return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.warped_src:warped_src, self.warped_dst:warped_dst, self.morph_value_t:[morph_value] })

            self.AE_view = AE_view
        else:
            #Initializing merge function
            with tf.device( nn.tf_default_device_name if len(devices) != 0 else f'/CPU:0'):
                gpu_dst_code = self.encoder (self.warped_dst)
                gpu_dst_inter_src_code = self.inter_src (gpu_dst_code)
                gpu_dst_inter_dst_code = self.inter_dst (gpu_dst_code)

                inter_dims_slice = tf.cast(inter_dims*self.morph_value_t[0], tf.int32)
                gpu_src_dst_code =  tf.concat( ( tf.slice(gpu_dst_inter_src_code, [0,0,0,0],   [-1, inter_dims_slice , inter_res, inter_res]),
                                                 tf.slice(gpu_dst_inter_dst_code, [0,inter_dims_slice,0,0], [-1,inter_dims-inter_dims_slice, inter_res,inter_res]) ), 1 )

                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                _, gpu_pred_dst_dstm = self.decoder(gpu_dst_inter_dst_code)

            def AE_merge(warped_dst, morph_value):
                return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst, self.morph_value_t:[morph_value] })

            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            do_init = self.is_first_run()
            if self.is_training and gan_power != 0 and model == self.GAN:
                if self.gan_model_changed:
                    do_init = True
            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )
            if do_init:
                model.init_weights()
        ###############

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path #if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path #if not self.pretrain else self.get_pretraining_data_path()

            random_ct_samples_path=training_data_dst_path if ct_mode is not None else None #and not self.pretrain

            cpu_count = multiprocessing.cpu_count()
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)



            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.125, 0.125], random_flip=self.random_src_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                         'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                         'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,  'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'],# or self.pretrain,
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.125, 0.125], random_flip=self.random_dst_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                             'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                             'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,  'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'],# or self.pretrain,
                        generators_count=dst_generators_count )
                             ])

    def export_dfm (self):
        output_path=self.get_strpath_storage_for_file('model.dfm')

        io.log_info(f'Dumping .dfm to {output_path}')

        tf = nn.tf
        with tf.device (nn.tf_default_device_name):
            warped_dst = tf.placeholder (nn.floatx, (None, self.resolution, self.resolution, 3), name='in_face')
            warped_dst = tf.transpose(warped_dst, (0,3,1,2))
            morph_value = tf.placeholder (nn.floatx, (1,), name='morph_value')

            gpu_dst_code = self.encoder (warped_dst)
            gpu_dst_inter_src_code = self.inter_src ( gpu_dst_code)
            gpu_dst_inter_dst_code = self.inter_dst ( gpu_dst_code)

            inter_dims_slice = tf.cast(self.inter_dims*morph_value[0], tf.int32)
            gpu_src_dst_code =  tf.concat( (tf.slice(gpu_dst_inter_src_code, [0,0,0,0],   [-1, inter_dims_slice , self.inter_res, self.inter_res]),
                                            tf.slice(gpu_dst_inter_dst_code, [0,inter_dims_slice,0,0], [-1,self.inter_dims-inter_dims_slice, self.inter_res,self.inter_res]) ), 1 )

            gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
            _, gpu_pred_dst_dstm = self.decoder(gpu_dst_inter_dst_code)

            gpu_pred_src_dst = tf.transpose(gpu_pred_src_dst, (0,2,3,1))
            gpu_pred_dst_dstm = tf.transpose(gpu_pred_dst_dstm, (0,2,3,1))
            gpu_pred_src_dstm = tf.transpose(gpu_pred_src_dstm, (0,2,3,1))

        tf.identity(gpu_pred_dst_dstm, name='out_face_mask')
        tf.identity(gpu_pred_src_dst, name='out_celeb_face')
        tf.identity(gpu_pred_src_dstm, name='out_celeb_face_mask')

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            nn.tf_sess,
            tf.get_default_graph().as_graph_def(),
            ['out_face_mask','out_celeb_face','out_celeb_face_mask']
        )

        import tf2onnx
        with tf.device("/CPU:0"):
            model_proto, _ = tf2onnx.convert._convert_common(
                output_graph_def,
                name='AMP',
                input_names=['in_face:0','morph_value:0'],
                output_names=['out_face_mask:0','out_celeb_face:0','out_celeb_face_mask:0'],
                opset=9,
                output_path=output_path)

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def should_save_preview_history(self):
        return (not io.is_colab() and self.iter % ( 10*(max(1,self.resolution // 64)) ) == 0) or \
               (io.is_colab() and self.iter % 100 == 0)

    #override
    def onTrainOneIter(self):
        bs = self.get_batch_size()

        ( (warped_src, target_src, target_srcm, target_srcm_em), \
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = self.generate_next_samples()

        src_loss, dst_loss = self.train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        if self.gan_power != 0:
            self.GAN_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )

    #override
    def onGetPreview(self, samples, for_history=False):
        ( (warped_src, target_src, target_srcm, target_srcm_em),
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = samples

        S, D, SS, DD, DDM_000, _, _ = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst, 0.0)  ) ]

        _, _, DDM_025, SD_025, SDM_025 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.25) ]
        _, _, DDM_050, SD_050, SDM_050 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.50) ]
        _, _, DDM_065, SD_065, SDM_065 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.65) ]
        _, _, DDM_075, SD_075, SDM_075 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.75) ]
        _, _, DDM_100, SD_100, SDM_100 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 1.00) ]

        (DDM_000,
         DDM_025, SDM_025,
         DDM_050, SDM_050,
         DDM_065, SDM_065,
         DDM_075, SDM_075,
         DDM_100, SDM_100) = [ np.repeat (x, (3,), -1) for x in (DDM_000,
                                                                 DDM_025, SDM_025,
                                                                 DDM_050, SDM_050,
                                                                 DDM_065, SDM_065,
                                                                 DDM_075, SDM_075,
                                                                 DDM_100, SDM_100) ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        result = []

        i = np.random.randint(n_samples) if not for_history else 0

        st =  [ np.concatenate ((S[i],  D[i],  DD[i]*DDM_000[i]), axis=1) ]
        st += [ np.concatenate ((SS[i], DD[i], SD_100[i] ), axis=1) ]

        result += [ ('AMP morph 1.0', np.concatenate (st, axis=0 )), ]

        st =  [ np.concatenate ((DD[i], SD_025[i],  SD_050[i]), axis=1) ]
        st += [ np.concatenate ((SD_065[i], SD_075[i], SD_100[i]), axis=1) ]
        result += [ ('AMP morph list', np.concatenate (st, axis=0 )), ]

        st =  [ np.concatenate ((DD[i], SD_025[i]*DDM_025[i]*SDM_025[i],  SD_050[i]*DDM_050[i]*SDM_050[i]), axis=1) ]
        st += [ np.concatenate ((SD_065[i]*DDM_065[i]*SDM_065[i], SD_075[i]*DDM_075[i]*SDM_075[i], SD_100[i]*DDM_100[i]*SDM_100[i]), axis=1) ]
        result += [ ('AMP morph list masked', np.concatenate (st, axis=0 )), ]

        return result

    def predictor_func (self, face, morph_value):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face, morph_value) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        morph_factor = np.clip ( io.input_number ("Morph factor", 1.0, add_info="0.0 .. 1.0"), 0.0, 1.0 )

        def predictor_morph(face):
            return self.predictor_func(face, morph_factor)


        import merger
        return predictor_morph, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = AMPModel
