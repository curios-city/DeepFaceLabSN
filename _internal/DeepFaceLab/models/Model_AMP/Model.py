import multiprocessing
import operator

import numpy as np

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *
from core.cv2ex import *

from pathlib import Path

from utils.label_face import label_face_filename

from utils.train_status_export import data_format_change, prepare_sample
import cv2
from core.cv2ex import cv2_imwrite
from tqdm import tqdm

class AMPModel(ModelBase):

    #override
    def on_initialize_options(self):
        default_retraining_samples = self.options['retraining_samples'] = self.load_or_def_option('retraining_samples', False)
        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 224)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'f')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_inter_dims         = self.options['inter_dims']         = self.load_or_def_option('inter_dims', 1024)

        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)

        default_morph_factor       = self.options['morph_factor']       = self.load_or_def_option('morph_factor', 0.5)
        default_preview_mf         = self.options['preview_mf']         = self.load_or_def_option('preview_mf', 1)

        default_masked_training    = self.options['masked_training']    = self.load_or_def_option('masked_training', True)

        default_eyes_prio          = self.options['eyes_prio']          = self.load_or_def_option('eyes_prio', False)
        default_mouth_prio         = self.options['mouth_prio']         = self.load_or_def_option('mouth_prio', False)

        # Compatibility check
        eyes_mouth_prio = self.options.get('eyes_mouth_prio')
        if eyes_mouth_prio is not None:
            default_eyes_prio = self.options['eyes_prio'] = eyes_mouth_prio
            default_mouth_prio = self.options['mouth_prio'] = eyes_mouth_prio
            self.options.pop('eyes_mouth_prio')

        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)

        # Uncomment it just if you want to impelement other loss functions
        default_loss_function      = self.options['loss_function']      = self.load_or_def_option('loss_function', 'SSIM')

        default_blur_out_mask      = self.options['blur_out_mask']      = self.load_or_def_option('blur_out_mask', False)

        default_adabelief          = self.options['adabelief']          = self.load_or_def_option('adabelief', True)

        default_lr_dropout         = self.options['lr_dropout']         = self.load_or_def_option('lr_dropout', 'n')

        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_random_hsv_power   = self.options['random_hsv_power']   = self.load_or_def_option('random_hsv_power', 0.0)
        default_random_downsample  = self.options['random_downsample']  = self.load_or_def_option('random_downsample', False)
        default_random_noise       = self.options['random_noise']       = self.load_or_def_option('random_noise', False)
        default_random_blur        = self.options['random_blur']        = self.load_or_def_option('random_blur', False)
        default_random_jpeg        = self.options['random_jpeg']        = self.load_or_def_option('random_jpeg', False)

        #random_shadow_src_options = self.options['random_shadow_src']   = self.load_or_def_option('random_shadow_src', False)
        #random_shadow_dst_options = self.options['random_shadow_dst']   = self.load_or_def_option('random_shadow_dst', False)

        if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:

            if isinstance(random_shadow_src_options, list) and isinstance(random_shadow_dst_options, list):
                for opt in random_shadow_src_options:
                    if 'enabled' in opt.keys():
                        random_shadow_src = opt['enabled']
                for opt in random_shadow_dst_options:
                    if 'enabled' in opt.keys():
                        random_shadow_dst = opt['enabled']

                if random_shadow_src and random_shadow_dst:
                    self.options['random_shadow'] = 'all'
                elif random_shadow_src:
                    self.options['random_shadow'] = 'src'
                elif random_shadow_dst:
                    self.options['random_shadow'] = 'dst'
                else:
                    self.options['random_shadow'] = 'none'
                default_random_shadow = self.load_or_def_option('random_shadow', 'none')
            else:
                default_random_shadow = self.options['random_shadow'] = self.load_or_def_option('random_shadow', 'none')

            del self.options['random_shadow_src']
            del self.options['random_shadow_dst']

        # Uncomment it just if you want to impelement other loss functions
        default_background_power   = self.options['background_power']   = self.load_or_def_option('background_power', 0.0)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_random_color       = self.options['random_color']       = self.load_or_def_option('random_color', False)
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)
        default_usefp16            = self.options['use_fp16']           = self.load_or_def_option('use_fp16', False)
        default_cpu_cap            = self.options['cpu_cap']            = self.load_or_def_option('cpu_cap', 8)
        default_preview_samples    = self.options['preview_samples']    = self.load_or_def_option('preview_samples', 4)
        default_full_preview       = self.options['force_full_preview'] = self.load_or_def_option('force_full_preview', False)
        default_lr                 = self.options['lr']                 = self.load_or_def_option('lr', 5e-5)

        ask_override = False if self.read_from_conf else self.ask_override()
        if self.is_first_run() or ask_override:
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                self.ask_autobackup_hour()
                self.ask_maximum_n_backups()
                self.ask_write_preview_history()
                self.options['preview_samples'] = np.clip ( io.input_int ("预览样本数量", default_preview_samples, add_info="1 - 16", help_message="典型的精细值为4"), 1, 16 )
                self.options['force_full_preview'] = io.input_bool ("使用旧的预览面板", default_full_preview)


                self.ask_target_iter()
                self.ask_retraining_samples(default_retraining_samples)
                self.ask_random_src_flip()
                self.ask_random_dst_flip()
                self.ask_batch_size(8)
                self.options['use_fp16'] = io.input_bool ("使用fp16", default_usefp16, help_message='提高训练/推理速度，缩小模型大小。模型可能会崩溃。1-5k 迭代次数后启用.')
                self.options['cpu_cap'] = np.clip ( io.input_int ("最大使用的 CPU 核心数.", default_cpu_cap, add_info="1 - 256", help_message="典型的精细值为 8"), 1, 256 )



        if self.is_first_run():
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                resolution = io.input_int("分辨率 Resolution", default_resolution, add_info="64-640", help_message="更高的分辨率需要更多的 VRAM 和训练时间。该值将调整为 16 和 32 的倍数，以适应不同的架构.")
                resolution = np.clip ( (resolution // 32) * 32, 64, 640)
                self.options['resolution'] = resolution
                self.options['face_type'] = io.input_str ("人脸类型 Face type", default_face_type, ['h','mf','f','wf','head', 'custom'], help_message="Half / mid face / full face / whole face / head / custom. 半脸/中脸/全脸/全脸/头部/自定义。半脸的分辨率较高，但覆盖脸颊的面积较小。中脸比半脸宽 30%。全脸 包括前额在内的整个脸部。头部覆盖整个头部，但需要 XSeg 来获取源和目的面部集.").lower()



        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', 64)

        default_d_mask_dims        = default_d_dims // 3
        default_d_mask_dims        += default_d_mask_dims % 2
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                self.options['ae_dims']    = np.clip ( io.input_int("自动编码器尺寸 AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="所有面部信息将被打包到 AE 维度中。如果 AE 维度的数量不足，例如闭眼可能无法被识别。维度越多越好，但需要更多的显存。您可以微调模型大小以适应您的 GPU" ), 32, 1024 )
                self.options['inter_dims'] = np.clip ( io.input_int("内部维度 Inter dimensions", default_inter_dims, add_info="32-2048", help_message="应等于或大于自动编码器尺寸。尺寸越大越好，但需要更多的 VRAM。您可以微调模型尺寸以适应您的 GPU." ), 32, 2048 )

                e_dims = np.clip ( io.input_int("编码器尺寸 Encoder dimensions", default_e_dims, add_info="16-256", help_message="更多的维度有助于识别更多的面部特征并获得更清晰的效果，但需要更多的 VRAM。您可以微调模型大小以适应您的 GPU." ), 16, 256 )
                self.options['e_dims'] = e_dims + e_dims % 2

                d_dims = np.clip ( io.input_int("解码器尺寸 Decoder dimensions", default_d_dims, add_info="16-256", help_message="更多的维度有助于识别更多的面部特征并获得更清晰的效果，但需要更多的 VRAM。您可以微调模型大小以适应您的 GPU." ), 16, 256 )
                self.options['d_dims'] = d_dims + d_dims % 2

                d_mask_dims = np.clip ( io.input_int("解码器掩码尺寸 Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="典型的掩码尺寸 = 解码器尺寸 / 3。 如果手动从 dst 掩码中剪除障碍物，可以增加该参数以获得更好的质量" ), 16, 256 )
                self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

        if self.is_first_run() or ask_override:
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                morph_factor = np.clip ( io.input_number ("变形因子 Morph factor.", default_morph_factor, add_info="0.1 .. 0.5", help_message="典型的精细值为 0.5"), 0.1, 0.5 )
                self.options['morph_factor'] = morph_factor

                preview_mf = io.input_number ("预览变形因子 Preview morph factor.", default_preview_mf, add_info="0.25 | 0.50 | 0.65 | 0.75 | 1", valid_list=[0.25, 0.50, 0.65, 0.75, 1], help_message="预览中最后一列的变形因子1/3")
                self.options['preview_mf'] = preview_mf

                if self.options['face_type'] == 'wf' or self.options['face_type'] == 'head':
                        self.options['masked_training']  = io.input_bool ("遮罩训练 Masked training", default_masked_training, help_message="此选项仅适用于wf或head类型。遮罩训练将训练区域剪辑为全脸遮罩或 XSeg 遮罩，这样网络就能正确地训练人脸")

                self.options['eyes_prio'] = io.input_bool ("眼睛优先 Eyes priority", default_eyes_prio, help_message='通过强制神经网络优先训练眼部，有助于在训练过程中修复眼睛问题，如外星人眼和错误的眼睛方向，特别是在高清架构上，在此之前/之后 https://i.imgur.com/YQHOuSR.jpg ')
                self.options['mouth_prio'] = io.input_bool ("嘴巴优先 Mouth priority", default_mouth_prio, help_message='通过强制神经网络优先训练嘴部，类似于眼睛，有助于在训练过程中修复嘴巴问题')

                self.options['uniform_yaw'] = io.input_bool ("侧脸优化 Uniform yaw distribution of samples", default_uniform_yaw, help_message='有助于修复由于人脸数据中的侧脸数量较少而导致的模糊问题')

                self.options['blur_out_mask'] = io.input_bool ("遮罩边缘模糊 Blur out mask", default_blur_out_mask, help_message='模糊训练样本中应用的脸部遮罩之外的附近区域。其结果是，脸部附近的背景被平滑化，在交换的脸部上不那么明显。需要在 src 和 dst 脸集中使用精确的 xseg 遮罩')

                self.options['loss_function'] = io.input_str(f"损失函数 Loss function", default_loss_function, ['SSIM', 'MS-SSIM', 'MS-SSIM+L1'], help_message="用于图像质量评估的变化损失函数")
                self.options['lr'] = np.clip (io.input_number("学习率 Learning rate", default_lr, add_info="0.0 .. 1.0", help_message="学习率：典型精细值 5e-5"), 0.0, 1)

                self.options['lr_dropout']  = io.input_str (f"使用学习率下降 Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="当人脸训练得足够好时，可以启用该选项来获得额外的清晰度，并减少子像素抖动，从而减少迭代次数。在 禁用随机扭曲 和 GAN 之前启用。在 CPU 上启用。这样就可以不使用额外的 VRAM，牺牲 20% 的迭代时间")

        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size     = self.options['gan_patch_size']     = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
        default_gan_dims           = self.options['gan_dims']           = self.load_or_def_option('gan_dims', 16)
        default_gan_smoothing      = self.options['gan_smoothing']      = self.load_or_def_option('gan_smoothing', 0.1)
        default_gan_noise          = self.options['gan_noise']          = self.load_or_def_option('gan_noise', 0.0)

        if self.is_first_run() or ask_override:
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                self.options['models_opt_on_gpu'] = io.input_bool ("将模型和优化器放到GPU上 Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="在一个 GPU 上进行训练时，默认情况下模型和优化器权重会放在 GPU 上，以加速训练过程。您可以将它们放在 CPU 上，以释放额外的 VRAM，从而设置更大的维度.")

                self.options['adabelief'] = io.input_bool ("使用AdaBelief优化器 Use AdaBelief optimizer?", default_adabelief, help_message="使用 AdaBelief 优化器。它需要更多的 VRAM，但模型的准确性和泛化程度更高.")

                self.options['random_warp'] = io.input_bool ("启用样本随机扭曲 Enable random warp of samples", default_random_warp, help_message="要概括两张人脸的面部表情，需要使用随机翘曲。当人脸训练得足够好时，可以禁用它来获得额外的清晰度，并减少亚像素抖动，从而减少迭代次数.")
                self.options['random_downsample'] = io.input_bool("启用样本随机采样降低采样率 Enable random downsample of samples", default_random_downsample, help_message="通过缩小部分样本来挑战模型")
                self.options['random_noise'] = io.input_bool("启用在样本中随机添加噪音 Enable random noise added to samples", default_random_noise, help_message="通过在某些样本中添加噪音来挑战模型")
                self.options['random_blur'] = io.input_bool("启用对样本的随机模糊 Enable random blur of samples", default_random_blur, help_message="通过在某些样本中添加模糊效果来挑战模型")
                self.options['random_jpeg'] = io.input_bool("启用随机压缩jpeg样本 Enable random jpeg compression of samples", default_random_jpeg, help_message="通过对某些样本应用 jpeg 压缩的质量降级来挑战模型")
                #self.options['random_shadow'] = io.input_str('启用对样本的随机阴影和高光 Enable random shadows and highlights of samples', default_random_shadow, ['none','src','dst','all'], help_message="有助于在数据集中创建暗光区域。如果你的src数据集缺乏阴影/不同的光照情况；使用dst以帮助泛化；或者使用all以满足两者的需求")
                self.options['random_hsv_power'] = np.clip ( io.input_number ("随机色调/饱和度/光强度 Random hue/saturation/light intensity", default_random_hsv_power, add_info="0.0 .. 0.3", help_message="随机色调/饱和度/光照强度仅应用于神经网络输入的src人脸集。稳定人脸交换过程中的色彩扰动。通过选择原始面孔集中最接近的面孔来降低色彩转换的质量。因此src人脸集必须足够多样化。典型的精细值为 0.05"), 0.0, 0.3 )

                self.options['gan_power'] = np.clip ( io.input_number ("GAN强度 GAN power", default_gan_power, add_info="0.0 .. 5.0", help_message="以生成对抗方式训练网络。强制神经网络学习人脸的小细节。只有当人脸训练得足够好时才启用它，否则就不要禁用。典型值为 0.1"), 0.0, 5.0 )


                if self.options['gan_power'] != 0.0:

                    gan_patch_size = np.clip ( io.input_int("GAN补丁大 GAN patch size", default_gan_patch_size, add_info="3-640", help_message="补丁大小越大，质量越高，需要的显存越多。即使在最低设置下，您也可以获得更清晰的边缘。典型的良好数值是分辨率除以8" ), 3, 640 )
                    self.options['gan_patch_size'] = gan_patch_size

                    gan_dims = np.clip ( io.input_int("GAN维度 GAN dimensions", default_gan_dims, add_info="4-64", help_message="GAN 网络的尺寸。尺寸越大，所需的 VRAM 越多。即使在最低设置下，也能获得更清晰的边缘。典型的精细值为 16" ), 4, 64 )
                    self.options['gan_dims'] = gan_dims

                    self.options['gan_smoothing'] = np.clip ( io.input_number("GAN标签平滑 GAN label smoothing", default_gan_smoothing, add_info="0 - 0.5", help_message="使用软标签，其值略微偏离 GAN 的 0/1，具有正则化效果"), 0, 0.5)
                    self.options['gan_noise'] = np.clip ( io.input_number("GAN噪声标签 GAN noisy labels", default_gan_noise, add_info="0 - 0.5", help_message="用错误的标签标记某些图像，有助于防止塌陷"), 0, 0.5)


                self.options['background_power'] = np.clip ( io.input_number("背景强度 Background power", default_background_power, add_info="0.0..1.0", help_message="了解遮罩外的区域。有助于平滑遮罩边界附近的区域。可随时使用"), 0.0, 1.0 )


                self.options['ct_mode'] = io.input_str (f"色彩转换模式 Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot', 'fs-aug', 'cc-aug'], help_message="改变靠近 dst 样本的 src 样本的颜色分布。尝试所有模式，找出最佳方案。FS aug 为 dst 和 sr 添加随机颜色")
                self.options['random_color'] = io.input_bool ("随机颜色 Random color", default_random_color, help_message="在LAB色彩空间中，样本随机围绕 L 轴旋转，有助于训练泛化")

                self.options['clipgrad'] = io.input_bool ("启用梯度裁剪 Enable gradient clipping", default_clipgrad, help_message="梯度裁剪降低了模型崩溃的几率，但牺牲了训练速度.")

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
        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'custom' : FaceType.CUSTOM,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]
        morph_factor = self.options['morph_factor']
        gan_power    = self.gan_power = self.options['gan_power']
        random_warp  = self.options['random_warp']
        random_hsv_power = self.options['random_hsv_power']

        if 'eyes_mouth_prio' in self.options:
            self.options.pop('eyes_mouth_prio')

        bg_factor = self.options['background_power']

        eyes_prio = self.options['eyes_prio']
        mouth_prio = self.options['mouth_prio']
        masked_training = self.options['masked_training']
        blur_out_mask = self.options['blur_out_mask'] if masked_training else False

        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        adabelief = self.options['adabelief']

        use_fp16 = self.options['use_fp16']
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
                if gan_power != 0:
                    self.GAN = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=input_ch, base_ch=self.options['gan_dims'], use_fp16=self.options['use_fp16'], name="D_src")
                    self.model_filename_list += [ [self.GAN, 'GAN.npy'] ]

                # Initialize optimizers
                lr = self.options['lr']

                clipnorm = 1.0 if self.options['clipgrad'] else 0.0
                if self.options['lr_dropout'] in ['y','cpu']:
                    lr_cos = 500
                    lr_dropout = 0.3
                else:
                    lr_cos = 0
                    lr_dropout = 1.0
                self.G_weights = self.encoder.get_weights() + self.decoder.get_weights()

                OptimizerClass = nn.AdaBelief if adabelief else nn.RMSprop
                self.src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, clipnorm=clipnorm, name='src_dst_opt')
                self.src_dst_opt.initialize_variables (self.G_weights, vars_on_cpu=optimizer_vars_on_cpu)
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

                if gan_power != 0:
                    self.GAN_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
                    self.GAN_opt.initialize_variables ( self.GAN.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')#+self.D_src_x2.get_weights()
                    self.model_filename_list += [ (self.GAN_opt, 'GAN_opt.npy') ]

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

            def DLoss(labels,logits):
                return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=[1,2,3])

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
                        gpu_target_srcm_all     = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_srcm_em  = self.target_srcm_em[batch_slice,:,:,:]
                        gpu_target_dstm_all     = self.target_dstm[batch_slice,:,:,:]
                        gpu_target_dstm_em  = self.target_dstm_em[batch_slice,:,:,:]

                    gpu_target_srcm_anti = 1-gpu_target_srcm_all
                    gpu_target_dstm_anti = 1-gpu_target_dstm_all

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


                    if blur_out_mask:
                        sigma = resolution / 128

                        x = nn.gaussian_blur(gpu_target_src*gpu_target_srcm_anti, sigma)
                        y = 1-nn.gaussian_blur(gpu_target_srcm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_src = gpu_target_src*gpu_target_srcm_all + (x/y)*gpu_target_srcm_anti

                        x = nn.gaussian_blur(gpu_target_dst*gpu_target_dstm_anti, sigma)
                        y = 1-nn.gaussian_blur(gpu_target_dstm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_dst = gpu_target_dst*gpu_target_dstm_all + (x/y)*gpu_target_dstm_anti

                    # unpack masks from one combined mask
                    gpu_target_srcm      = tf.clip_by_value (gpu_target_srcm_all, 0, 1)
                    gpu_target_dstm      = tf.clip_by_value (gpu_target_dstm_all, 0, 1)
                    gpu_target_srcm_eye_mouth = tf.clip_by_value (gpu_target_srcm_em-1, 0, 1)
                    gpu_target_dstm_eye_mouth = tf.clip_by_value (gpu_target_dstm_em-1, 0, 1)
                    gpu_target_srcm_mouth = tf.clip_by_value (gpu_target_srcm_em-2, 0, 1)
                    gpu_target_dstm_mouth = tf.clip_by_value (gpu_target_dstm_em-2, 0, 1)
                    gpu_target_srcm_eyes = tf.clip_by_value (gpu_target_srcm_eye_mouth-gpu_target_srcm_mouth, 0, 1)
                    gpu_target_dstm_eyes = tf.clip_by_value (gpu_target_dstm_eye_mouth-gpu_target_dstm_mouth, 0, 1)



                    gpu_target_srcm_gblur = nn.gaussian_blur(gpu_target_srcm, resolution // 32)
                    gpu_target_dstm_gblur = nn.gaussian_blur(gpu_target_dstm, resolution // 32)


                    gpu_target_srcm_blur = tf.clip_by_value(gpu_target_srcm_gblur, 0, 0.5) * 2
                    gpu_target_dstm_blur = tf.clip_by_value(gpu_target_dstm_gblur, 0, 0.5) * 2

                    gpu_target_srcm_anti_blur = 1.0-gpu_target_srcm_blur
                    gpu_target_dstm_anti_blur = 1.0-gpu_target_dstm_blur

                    gpu_target_src_masked = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked = gpu_target_dst*gpu_target_dstm_blur if masked_training else gpu_target_dst
                    gpu_target_src_anti_masked = gpu_target_src*gpu_target_srcm_anti_blur
                    gpu_target_dst_anti_masked = gpu_target_dst*gpu_target_dstm_anti_blur

                    gpu_pred_src_src_masked = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst
                    gpu_pred_src_src_anti_masked = gpu_pred_src_src*gpu_target_srcm_anti_blur
                    gpu_pred_dst_dst_anti_masked = gpu_pred_dst_dst*gpu_target_dstm_anti_blur

                    if self.options['loss_function'] == 'MS-SSIM':
                        gpu_src_loss =  10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_src_masked, gpu_pred_src_src_masked, max_val=1.0)
                        gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked - gpu_pred_src_src_masked ), axis=[1,2,3])
                        gpu_dst_loss  =  10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_dst_masked, gpu_pred_dst_dst_masked, max_val=1.0)
                        gpu_dst_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_dst_masked - gpu_pred_dst_dst_masked ), axis=[1,2,3])

                        if bg_factor  > 0:
                            gpu_dst_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0)
                            gpu_dst_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_dst - gpu_pred_dst_dst ), axis=[1,2,3])
                            gpu_src_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_src, gpu_pred_src_src, max_val=1.0)
                            gpu_src_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_src - gpu_pred_src_src ), axis=[1,2,3])

                    elif self.options['loss_function'] == 'MS-SSIM+L1':

                        gpu_src_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_src_masked, gpu_pred_src_src_masked, max_val=1.0)
                        gpu_dst_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_dst_masked, gpu_pred_dst_dst_masked, max_val=1.0)

                        if bg_factor > 0:
                            gpu_dst_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0)
                            gpu_src_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_src, gpu_pred_src_src, max_val=1.0)

                    else:
                        gpu_src_loss =  tf.reduce_mean (5*nn.dssim(gpu_target_src_masked, gpu_pred_src_src_masked, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                        gpu_src_loss += tf.reduce_mean (5*nn.dssim(gpu_target_src_masked, gpu_pred_src_src_masked, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])

                        gpu_dst_loss =  tf.reduce_mean (5*nn.dssim(gpu_target_dst_masked, gpu_pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                        gpu_dst_loss += tf.reduce_mean (5*nn.dssim(gpu_target_dst_masked, gpu_pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/23.2) ), axis=[1])

                        # Pixel loss
                        gpu_dst_loss += tf.reduce_mean (10*tf.square(gpu_target_dst_masked-gpu_pred_dst_dst_masked), axis=[1,2,3])
                        gpu_src_loss += tf.reduce_mean (10*tf.square(gpu_target_src_masked-gpu_pred_src_src_masked), axis=[1,2,3])

                        if bg_factor > 0:
                            gpu_dst_loss +=  bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                            gpu_dst_loss += bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                            gpu_src_loss +=  bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_src, gpu_pred_src_src, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                            gpu_src_loss += bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_src, gpu_pred_src_src, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])

                    if bg_factor > 0:
                        gpu_dst_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_dst - gpu_pred_dst_dst ), axis=[1,2,3])
                        gpu_src_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_src - gpu_pred_src_src ), axis=[1,2,3])



                    # Eyes+mouth prio loss
                    # if eyes_mouth_prio:
                        # gpu_src_loss += tf.reduce_mean (300*tf.abs (gpu_target_src*gpu_target_srcm_em-gpu_pred_src_src*gpu_target_srcm_em), axis=[1,2,3])
                        # gpu_dst_loss += tf.reduce_mean (300*tf.abs (gpu_target_dst*gpu_target_dstm_em-gpu_pred_dst_dst*gpu_target_dstm_em), axis=[1,2,3])

                    if eyes_prio or mouth_prio:
                        if eyes_prio and mouth_prio:
                            gpu_target_part_mask_src = gpu_target_srcm_eye_mouth
                            gpu_target_part_mask_dst = gpu_target_dstm_eye_mouth
                        elif eyes_prio:
                            gpu_target_part_mask_src = gpu_target_srcm_eyes
                            gpu_target_part_mask_dst = gpu_target_dstm_eyes
                        elif mouth_prio:
                            gpu_target_part_mask_src = gpu_target_srcm_mouth
                            gpu_target_part_mask_dst = gpu_target_dstm_mouth

                        gpu_src_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_src*gpu_target_part_mask_src - gpu_pred_src_src*gpu_target_part_mask_src ), axis=[1,2,3])
                        gpu_dst_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_dst*gpu_target_part_mask_dst - gpu_pred_dst_dst*gpu_target_part_mask_dst ), axis=[1,2,3])

                    # Mask loss
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm_all - gpu_pred_src_srcm ),axis=[1,2,3] )
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm_all - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]
                    gpu_G_loss = gpu_src_loss + gpu_dst_loss
                    # dst-dst background weak loss
                    gpu_G_loss += tf.reduce_mean(25*tf.square(gpu_pred_dst_dst_anti_masked-gpu_target_dst_anti_masked),axis=[1,2,3] )
                    gpu_G_loss += 0.00001*nn.total_variation_mse(gpu_pred_dst_dst_anti_masked)
                    # src-src background weak loss
                    gpu_G_loss += tf.reduce_mean(25*tf.square(gpu_pred_src_src_anti_masked-gpu_target_src_anti_masked),axis=[1,2,3] )
                    gpu_G_loss += 0.00001*nn.total_variation_mse(gpu_pred_src_src_anti_masked)


                    if gan_power != 0:

                        gpu_pred_src_src_d, gpu_pred_src_src_d2 = self.GAN(gpu_pred_src_src_masked)

                        def get_smooth_noisy_labels(label, tensor, smoothing=0.1, noise=0.05):
                            num_labels = self.batch_size
                            for d in tensor.get_shape().as_list()[1:]:
                                num_labels *= d

                            probs = tf.math.log([[noise, 1-noise]]) if label == 1 else tf.math.log([[1-noise, noise]])
                            x = tf.random.categorical(probs, num_labels)
                            x = tf.cast(x, tf.float32)
                            x = tf.math.scalar_mul(1-smoothing, x)
                            # x = x + (smoothing/num_labels)
                            x = tf.reshape(x, (self.batch_size,) + tuple(tensor.get_shape().as_list()[1:]))
                            return x

                        smoothing = self.options['gan_smoothing']
                        noise = self.options['gan_noise']

                        gpu_pred_src_src_d_ones = tf.ones_like(gpu_pred_src_src_d)
                        gpu_pred_src_src_d2_ones = tf.ones_like(gpu_pred_src_src_d2)

                        gpu_pred_src_src_d_smooth_zeros = get_smooth_noisy_labels(0, gpu_pred_src_src_d, smoothing=smoothing, noise=noise)
                        gpu_pred_src_src_d2_smooth_zeros = get_smooth_noisy_labels(0, gpu_pred_src_src_d2, smoothing=smoothing, noise=noise)

                        gpu_target_src_d, gpu_target_src_d2 = self.GAN(gpu_target_src_masked)

                        gpu_target_src_d_smooth_ones = get_smooth_noisy_labels(1, gpu_target_src_d, smoothing=smoothing, noise=noise)
                        gpu_target_src_d2_smooth_ones = get_smooth_noisy_labels(1, gpu_target_src_d2, smoothing=smoothing, noise=noise)

                        gpu_GAN_loss = DLoss(gpu_target_src_d_smooth_ones, gpu_target_src_d) \
                                             + DLoss(gpu_pred_src_src_d_smooth_zeros, gpu_pred_src_src_d) \
                                             + DLoss(gpu_target_src_d2_smooth_ones, gpu_target_src_d2) \
                                             + DLoss(gpu_pred_src_src_d2_smooth_zeros, gpu_pred_src_src_d2)

                        gpu_GAN_loss_gradients += [ nn.gradients (gpu_GAN_loss, self.GAN.get_weights() ) ]

                        gpu_G_loss += gan_power*(DLoss(gpu_pred_src_src_d_ones, gpu_pred_src_src_d)  + \
                                                 DLoss(gpu_pred_src_src_d2_ones, gpu_pred_src_src_d2))

                        if masked_training:
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

            def get_src_dst_information(warped_src, target_src, target_srcm, target_srcm_em,  \
                                        warped_dst, target_dst, target_dstm, target_dstm_em, ):
                out_data =nn.tf_sess.run ( [ src_loss, dst_loss, pred_src_src, pred_src_srcm, pred_dst_dst,
                                            pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm:target_srcm,
                                                       self.target_srcm_em:target_srcm_em,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm:target_dstm,
                                                       self.target_dstm_em:target_dstm_em,
                                                       self.morph_value_t:[1.0]
                                                       })

                return out_data

            self.get_src_dst_information = get_src_dst_information

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
        for model, filename in io.progress_bar_generator(self.model_filename_list, "初始化模型"):
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
            ignore_same_path = False

            random_ct_samples_path=training_data_dst_path if ct_mode is not None else None #and not self.pretrain

            cpu_count = min(multiprocessing.cpu_count(), self.options['cpu_cap'])
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            # If conf file is not used or doesn't exist
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                random_shadow_src = True if self.options['random_shadow'] in ['all', 'src'] else False
                random_shadow_dst = True if self.options['random_shadow'] in ['all', 'dst'] else False

                # it means is the first time we create the model using conf file
                if not self.config_file_exists and self.read_from_conf:
                    del self.options['random_shadow']
            else:
                random_shadow_src = self.options['random_shadow_src']
                random_shadow_dst = self.options['random_shadow_dst']

            dst_aug = None
            allowed_dst_augs = ['fs-aug', 'cc-aug']
            if ct_mode in allowed_dst_augs:
                dst_aug = ct_mode

            channel_type = SampleProcessor.ChannelType.LAB_RAND_TRANSFORM if self.options['random_color'] else SampleProcessor.ChannelType.BGR

            # Check for pak names
            # give priority to pak names in configuration file
            if self.read_from_conf and self.config_file_exists:
                conf_src_pak_name = self.options.get('src_pak_name', None)
                conf_dst_pak_name = self.options.get('dst_pak_name', None)
                if conf_src_pak_name is not None:
                    self.src_pak_name = conf_src_pak_name
                if conf_dst_pak_name is not None:
                    self.dst_pak_name = conf_dst_pak_name

            if self.src_pak_name != self.dst_pak_name and training_data_src_path == training_data_dst_path:
                ignore_same_path = True

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, pak_name=self.src_pak_name, ignore_same_path=ignore_same_path,
                        random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.125, 0.125], random_flip=self.random_src_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp,
                                                 'random_downsample': self.options['random_downsample'],
                                                 'random_noise': self.options['random_noise'],
                                                 'random_blur': self.options['random_blur'],
                                                 'random_jpeg': self.options['random_jpeg'],
                                                 'random_shadow': random_shadow_src,
                                                 'transform':True, 'channel_type' : channel_type, 'ct_mode': ct_mode,
                                                 'random_hsv_shift_amount' : random_hsv_power,
                                                 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'random_hsv_shift_amount' : random_hsv_power,
                                                'transform':True, 'channel_type' : channel_type, 'ct_mode': ct_mode, 'random_shadow': random_shadow_src,
                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'], #or self.pretrain
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, pak_name=self.src_pak_name, ignore_same_path=ignore_same_path,
                        debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.125, 0.125], random_flip=self.random_dst_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp,
                                                 'random_downsample': self.options['random_downsample'],
                                                 'random_noise': self.options['random_noise'],
                                                 'random_blur': self.options['random_blur'],
                                                 'random_jpeg': self.options['random_jpeg'],
                                                 'random_shadow': random_shadow_dst,
                                                 'transform':True, 'channel_type' : channel_type, 'ct_mode': dst_aug,
                                                 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : channel_type, 'ct_mode': dst_aug, 'random_shadow': random_shadow_dst,   'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'], #or self.pretrain,
                        generators_count=dst_generators_count )
                             ])

            if self.options['retraining_samples']:
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

    def export_dfm (self):
        output_path=self.get_strpath_storage_for_file('model.dfm')

        io.log_info(f'导出 .dfm 到 {output_path}')

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
                opset=12,
                output_path=output_path)

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "保存中...", leave=False):
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

        if self.options['retraining_samples']:
            for i in range(bs):
                self.last_src_samples_loss.append ( (src_loss[i], target_src[i], target_srcm[i], target_srcm_em[i]) )
                self.last_dst_samples_loss.append ( (dst_loss[i], target_dst[i], target_dstm[i], target_dstm_em[i]) )

            if len(self.last_src_samples_loss) >= bs*16:
                src_samples_loss = sorted(self.last_src_samples_loss, key=operator.itemgetter(0), reverse=True)
                dst_samples_loss = sorted(self.last_dst_samples_loss, key=operator.itemgetter(0), reverse=True)

                target_src        = np.stack( [ x[1] for x in src_samples_loss[:bs] ] )
                target_srcm       = np.stack( [ x[2] for x in src_samples_loss[:bs] ] )
                target_srcm_em    = np.stack( [ x[3] for x in src_samples_loss[:bs] ] )

                target_dst        = np.stack( [ x[1] for x in dst_samples_loss[:bs] ] )
                target_dstm       = np.stack( [ x[2] for x in dst_samples_loss[:bs] ] )
                target_dstm_em    = np.stack( [ x[3] for x in dst_samples_loss[:bs] ] )

                src_loss, dst_loss = self.train (target_src, target_src, target_srcm, target_srcm_em, target_dst, target_dst, target_dstm, target_dstm_em)
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

        if self.gan_power != 0:
            self.GAN_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )

    #override
    def onGetPreview(self, samples, for_history=False, filenames=None):
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

        morphed_preview_dict = {
            0.25: SD_025,
            0.50: SD_050,
            0.65: SD_065,
            0.75: SD_075,
            1   : SD_100,
        }

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(self.get_batch_size(), self.options['preview_samples'])

        result = []
        if self.options['force_full_preview']:
            i = np.random.randint(n_samples) if not for_history else 0

            if filenames is not None and len(filenames) > 0:
                S[i] = label_face_filename(S[i], filenames[0][i])
                D[i] = label_face_filename(D[i], filenames[1][i])

            st =  [ np.concatenate ((S[i],  D[i],  DD[i]*DDM_000[i]), axis=1) ]
            st += [ np.concatenate ((SS[i], DD[i], morphed_preview_dict[self.options['preview_mf']][i] ), axis=1) ]

            result += [ ('AMP morph 0.75', np.concatenate (st, axis=0 )), ]

            st =  [ np.concatenate ((DD[i], SD_025[i],  SD_050[i]), axis=1) ]
            st += [ np.concatenate ((SD_065[i], SD_075[i], SD_100[i]), axis=1) ]
            result += [ ('AMP morph list', np.concatenate (st, axis=0 )), ]


            st =  [ np.concatenate ((DD[i], SD_025[i]*DDM_025[i]*SDM_025[i],  SD_050[i]*DDM_050[i]*SDM_050[i]), axis=1) ]
            st += [ np.concatenate ((SD_065[i]*DDM_065[i]*SDM_065[i], SD_075[i]*DDM_075[i]*SDM_075[i], SD_100[i]*DDM_100[i]*SDM_100[i]), axis=1) ]
            result += [ ('AMP morph list masked', np.concatenate (st, axis=0 )), ]

        else:
            for i in range(n_samples if not for_history else 1):
                if filenames is not None and len(filenames) > 0:
                    S[i] = label_face_filename(S[i], filenames[0][i])
                    D[i] = label_face_filename(D[i], filenames[1][i])
            st = []
            temp_r = []
            for i in range(n_samples if not for_history else 1):
                st =  [ np.concatenate ((S[i], SS[i],  D[i]), axis=1) ]
                st += [ np.concatenate ((DD[i], DD[i]*DDM_000[i], morphed_preview_dict[self.options['preview_mf']][i] ), axis=1) ]
                temp_r += [ np.concatenate (st, axis=1) ]
            result += [ ('AMP morph 1.0', np.concatenate (temp_r, axis=0 )), ]
            # result += [ ('AMP morph 1.0', np.concatenate (st, axis=0 )), ]
            st = []
            temp_r = []
            for i in range(n_samples if not for_history else 1):
                st =  [ np.concatenate ((DD[i], SD_025[i],  SD_050[i]), axis=1) ]
                st += [ np.concatenate ((SD_065[i], SD_075[i], SD_100[i]), axis=1) ]
                temp_r += [ np.concatenate (st, axis=1) ]
            result += [ ('AMP morph list', np.concatenate (temp_r, axis=0 )), ]
            #result += [ ('AMP morph list', np.concatenate (st, axis=0 )), ]
            st = []
            temp_r = []
            for i in range(n_samples if not for_history else 1):
                st = [ np.concatenate ((DD[i], SD_025[i]*DDM_025[i]*SDM_025[i],  SD_050[i]*DDM_050[i]*SDM_050[i]), axis=1) ]
                st += [ np.concatenate ((SD_065[i]*DDM_065[i]*SDM_065[i], SD_075[i]*DDM_075[i]*SDM_075[i], SD_100[i]*DDM_100[i]*SDM_100[i]), axis=1) ]
                temp_r += [ np.concatenate (st, axis=1) ]
            result += [ ('AMP morph list masked', np.concatenate (temp_r, axis=0 )), ]
            #result += [ ('AMP morph list masked', np.concatenate (st, axis=0 )), ]

        return result

    def predictor_func (self, face, morph_value):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face, morph_value) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):

        def predictor_morph(face, func_morph_factor=1.0):
            return self.predictor_func(face, func_morph_factor)

        import merger
        return predictor_morph, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay', is_morphable=True)

    #override
    def get_config_schema_path(self):
        config_path = Path(__file__).parent.absolute() / Path("config_schema.json")
        return config_path

    #override
    def get_formatted_configuration_path(self):
        config_path = Path(__file__).parent.absolute() / Path("formatted_config.yaml")
        return config_path

    # function is WIP
    def generate_training_state(self):
        from tqdm import tqdm

        import datetime
        import json
        from itertools import zip_longest
        import multiprocessing as mp


        src_gen = self.generator_list[0]
        dst_gen =  self.generator_list[1]
        self.src_sample_state = []
        self.dst_sample_state = []

        src_samples = src_gen.samples
        dst_samples = dst_gen.samples
        src_len = len(src_samples)
        dst_len = len(dst_samples)
        length = src_len
        if length < dst_len:
            length = dst_len

        # set paths
        # create core folder
        self.state_history_path = self.saved_models_path / ( f'{self.get_model_name()}_state_history' )
        if not self.state_history_path.exists():
            self.state_history_path.mkdir(exist_ok=True)
        # create state folder
        idx_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        idx_state_history_path= self.state_history_path / idx_str
        idx_state_history_path.mkdir()
        # create set folders
        self.src_state_path = idx_state_history_path / 'src'
        self.src_state_path.mkdir()
        self.dst_state_path = idx_state_history_path / 'dst'
        self.dst_state_path.mkdir()

        print ('Generating dataset state snapshot\r')

        # doing batch 2 always since it is coded to always expect dst and src
        # if one is smaller reapeating the last sample as a placeholder

        # 0 means ignore and use dummy data
        data_list = list(zip_longest(src_samples, dst_samples, fillvalue=0))
        self._dummy_input = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
        self._dummy_mask = np.zeros((self.resolution, self.resolution, 1), dtype=np.float32)

        for sample_tuple in tqdm(data_list, desc='Processing samples', total=len(data_list)):
            self._processor(sample_tuple)

        # save model state params
        # copy model summary
        # model_summary = self.options.copy()
        model_summary = {}
        model_summary['iter'] = self.get_iter()
        model_summary['name'] = self.get_model_name()

        # error with some types, need to double check
        with open(idx_state_history_path / 'model_summary.json', 'w') as outfile:
            json.dump(model_summary, outfile)

        # training state, full loss stuff from .dat file - prolly should be global
        # state_history_json = self.loss_history

        # main config data
        # set name and full path
        config_dict = {
            'datasets': [{'name': 'src', 'path': str(self.training_data_src_path) }, {'name': 'dst', 'path': str(self.training_data_dst_path) }]
        }
        with open(self.state_history_path / 'config.json', 'w') as outfile:
            json.dump(config_dict, outfile)

        print ('完成')

        # save image loss data
        src_full_state_dict = {
            'data': self.src_sample_state,
            'set': 'src',
            'type': 'set-state'
        }
        with open(idx_state_history_path / 'src_state.json', 'w') as outfile:
            json.dump(src_full_state_dict, outfile)

        dst_full_state_dict = {
            'data': self.dst_sample_state,
            'set': 'dst',
            'type': 'set-state'
        }
        with open(idx_state_history_path / 'dst_state.json', 'w') as outfile:
            json.dump(dst_full_state_dict, outfile)

        print ('完成')

    def _get_formatted_image(self, raw_output):
        formatted = np.clip( nn.to_data_format(raw_output,"NHWC", self.model_data_format), 0.0, 1.0)
        formatted = np.squeeze(formatted, 0)

        return formatted

    def _processor(self, samples_tuple):
        if samples_tuple[0] != 0:
            src_sample_bgr, src_sample_mask, src_sample_mask_em = prepare_sample(samples_tuple[0], self.options, self.resolution, self.face_type)
        else:
            src_sample_bgr, src_sample_mask, src_sample_mask_em = self._dummy_input, self._dummy_mask, self._dummy_mask
        if samples_tuple[1] != 0:
            dst_sample_bgr, dst_sample_mask, dst_sample_mask_em = prepare_sample(samples_tuple[1], self.options, self.resolution, self.face_type)
        else:
            dst_sample_bgr, dst_sample_mask, dst_sample_mask_em = self._dummy_input, self._dummy_mask, self._dummy_mask

        src_loss, dst_loss, pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm = self.get_src_dst_information(
            data_format_change(src_sample_bgr), data_format_change(src_sample_bgr), data_format_change(src_sample_mask),
            data_format_change(src_sample_mask_em), data_format_change(dst_sample_bgr), data_format_change(dst_sample_bgr),
            data_format_change(dst_sample_mask), data_format_change(dst_sample_mask_em))

        if samples_tuple[0] != 0:
            src_file_name = Path(samples_tuple[0].filename).stem

            cv2_imwrite(self.src_state_path / f"{src_file_name}_output.jpg", self._get_formatted_image(np.expand_dims(pred_src_src[0,:,:,:], axis=0)) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100 ] ) # output

            src_data = { 'loss': float(src_loss[0]), 'input': f"{src_file_name}.jpg", 'output': f"{src_file_name}_output.jpg" }
            self.src_sample_state.append(src_data)

        if samples_tuple[1] != 0:
            dst_file_name = Path(samples_tuple[1].filename).stem

            cv2_imwrite(self.dst_state_path / f"{dst_file_name}_output.jpg", self._get_formatted_image(pred_dst_dst) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100 ] ) # output
            cv2_imwrite(self.dst_state_path / f"{dst_file_name}_swap.jpg", self._get_formatted_image(pred_src_dst) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100 ] ) # swap

            dst_data = { 'loss': float(dst_loss[0]), 'input': f"{dst_file_name}.jpg", 'output': f"{dst_file_name}_output.jpg", 'swap': f"{dst_file_name}_swap.jpg"  }
            self.dst_sample_state.append(dst_data)

Model = AMPModel
