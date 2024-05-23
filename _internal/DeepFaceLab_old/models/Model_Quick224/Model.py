import multiprocessing  # 导入多进程模块
from functools import partial  # 导入partial函数，用于创建偏函数

import numpy as np  # 导入NumPy库

from core import mathlib  # 导入数学库
from core.interact import interact as io  # 从交互模块中导入interact函数，并重命名为io
from core.leras import nn  # 从leras模块中导入nn
from facelib import FaceType  # 导入FaceType枚举
from models import ModelBase  # 从models模块中导入ModelBase类
from samplelib import *  # 导入samplelib模块的所有内容

class QModel(ModelBase):
    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()  # 获取当前设备配置信息
        devices = device_config.devices  # 获取设备列表
        self.model_data_format = "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"  # 根据设备数量和调试模式设置数据格式
        nn.initialize(data_format=self.model_data_format)  # 初始化Neural Network，并设置数据格式
        tf = nn.tf  # 获取TensorFlow对象

        resolution = self.resolution = 224  # 设置分辨率为224
        self.face_type = FaceType.WHOLE_FACE  # 设置脸部类型为整脸
        ae_dims = 192  # 设置ae维度为192
        e_dims = 48  # 设置e维度为64
        d_dims = 48  # 设置d维度为64
        d_mask_dims = 16  # 设置d_mask维度为16
        self.pretrain = False  # 是否预训练为False
        self.pretrain_just_disabled = False  # 是否刚刚禁用预训练为False

        masked_training = True  # 是否进行遮罩训练为True

        models_opt_on_gpu = len(devices) >= 1 and all([dev.total_mem_gb >= 4 for dev in devices])  # 检查是否所有GPU都有足够的内存
        models_opt_device = nn.tf_default_device_name if models_opt_on_gpu and self.is_training else '/CPU:0'  # 设置优化器设备
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'  # 检查优化器变量是否在CPU上

        input_ch = 3  # 输入通道数为3
        bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)  # 获取BGR形状
        mask_shape = nn.get4Dshape(resolution,resolution,1)  # 获取掩码形状

        self.model_filename_list = []  # 模型文件名列表

        model_archi = nn.DeepFakeArchi(resolution, opts='d')  # 创建模型架构

        with tf.device ('/CPU:0'):
            # 在CPU上创建占位符
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape)
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_srcm = tf.placeholder (nn.floatx, mask_shape)
            self.target_dstm = tf.placeholder (nn.floatx, mask_shape)

        # 初始化模型类
        with tf.device (models_opt_device):
            self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')  # 创建编码器
            encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2  # 获取编码器输出通道数

            self.inter = model_archi.Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')  # 创建Inter模型
            inter_out_ch = self.inter.get_out_ch()  # 获取Inter模型输出通道数

            self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')  # 创建源解码器
            self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')  # 创建目标解码器

            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          [self.inter,       'inter.npy'      ],
                                          [self.decoder_src, 'decoder_src.npy'],
                                          [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()  # 获取可训练权重

                # 初始化优化器
                self.src_dst_opt = nn.RMSprop(lr=5e-5, lr_dropout=0.3, name='src_dst_opt')  # 创建优化器
                self.src_dst_opt.initialize_variables(self.src_dst_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu )  # 初始化变量
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]  # 将优化器文件名添加到列表中

        if self.is_training:
            # 调整每个GPU的批处理大小
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, 4 // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)  # 设置批处理大小

            # 计算每个GPU的损失
            gpu_pred_src_src_list = []
            gpu_pred_dst_dst_list = []
            gpu_pred_src_dst_list = []
            gpu_pred_src_srcm_list = []
            gpu_pred_dst_dstm_list = []
            gpu_pred_src_dstm_list = []

            gpu_src_losses = []
            gpu_dst_losses = []
            gpu_src_dst_loss_gvs = []
            
            for gpu_id in range(gpu_count):
                with tf.device( f'/{devices[gpu_id].tf_dev_type}:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
                    batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                    with tf.device(f'/CPU:0'):
                        # 在 CPU 上切片，否则所有批次数据将首先传输到 GPU
                        gpu_warped_src   = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst   = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src   = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst   = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm  = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_dstm  = self.target_dstm[batch_slice,:,:,:]

                    # 处理模型张量
                    gpu_src_code     = self.inter(self.encoder(gpu_warped_src))
                    gpu_dst_code     = self.inter(self.encoder(gpu_warped_dst))
                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )

                    gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur
                    gpu_target_dst_anti_masked = gpu_target_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_target_src_masked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst*gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

                    gpu_dst_loss  = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss
                    # 计算源目标损失和目标目标损失的梯度
                    gpu_src_dst_loss_gvs += [ nn.gradients ( gpu_G_loss, self.src_dst_trainable_weights ) ]


            # 平均损失和梯度，并创建优化器的更新操作
            with tf.device(models_opt_device):
                # 将GPU计算结果合并
                pred_src_src  = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst  = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst  = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

                # 计算平均损失
                src_loss = nn.average_tensor_list(gpu_src_losses)
                dst_loss = nn.average_tensor_list(gpu_dst_losses)
                src_dst_loss_gv = nn.average_gv_list(gpu_src_dst_loss_gvs)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op(src_dst_loss_gv)

            # 初始化训练和视图函数
            def src_dst_train(warped_src, target_src, target_srcm, \
                              warped_dst, target_dst, target_dstm):
                # 运行训练操作
                s, d, _ = nn.tf_sess.run([src_loss, dst_loss, src_dst_loss_gv_op],
                                          feed_dict={self.warped_src :warped_src,
                                                     self.target_src :target_src,
                                                     self.target_srcm:target_srcm,
                                                     self.warped_dst :warped_dst,
                                                     self.target_dst :target_dst,
                                                     self.target_dstm:target_dstm})
                # 计算平均损失
                s = np.mean(s)
                d = np.mean(d)
                return s, d

            self.src_dst_train = src_dst_train

            def AE_view(warped_src, warped_dst):
                # 运行视图操作
                return nn.tf_sess.run([pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                       feed_dict={self.warped_src:warped_src,
                                                  self.warped_dst:warped_dst})

            self.AE_view = AE_view

        else:
            # 初始化合并函数
            with tf.device(nn.tf_default_device_name if len(devices) != 0 else f'/CPU:0'):
                # 在指定设备上进行操作（如果有 GPU 设备，则使用 GPU，否则使用 CPU）
                gpu_dst_code = self.inter(self.encoder(self.warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            # 定义合并函数 AE_merge
            def AE_merge(warped_dst):
                # 运行合并函数，获取预测结果
                return nn.tf_sess.run([gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})

            # 将合并函数添加到类中
            self.AE_merge = AE_merge

        # 加载/初始化所有模型/优化器权重
        # 遍历模型及其对应的文件名列表，使用进度条显示加载状态
        for model, filename in io.progress_bar_generator(self.model_filename_list, "正在初始化模型"):
            # 默认不进行初始化
            do_init = False
            # 如果pretrain_just_disabled标志被设置，则仅对特定模型进行初始化
            if self.pretrain_just_disabled:
                #print('结束预训练,准备重置inter')
                if model == self.inter:
                    do_init = True
            else:
                # 如果是第一次运行，则需要进行初始化
                #print('模型首次运行')
                do_init = self.is_first_run()

            # 如果不需要进行初始化，则尝试从存储路径加载模型权重
            if not do_init:
                #print('不需要重置模型，正在加载本模型权重'+self.get_strpath_storage_for_file(filename))
                do_init = not model.load_weights(self.get_strpath_storage_for_file(filename))
                #print("本模型权重是否加载成功（相反）："+str(do_init))

            # 如果加载失败且存在预训练模型路径，则尝试从预训练路径加载权重
            if do_init and self.pretrained_model_path is not None:
                pretrained_filepath = self.pretrained_model_path / filename
                # 检查预训练文件路径是否存在
                if pretrained_filepath.exists():
                    #print('找到预置权重')
                    # 尝试加载预训练模型权重，如果失败则设置do_init为True进行初始化
                    do_init = not model.load_weights(pretrained_filepath)
                    #print("预置权重是否加载成功（相反）："+str(do_init)+str(pretrained_filepath))
            # 如果需要进行初始化，则调用模型的init_weights方法
            if do_init:
                #print('预置权重加载失败，强制重置模型')
                model.init_weights()


        # 初始化样本生成器
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True,  'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution}
                                              ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True,  'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution}
                                               ],
                        generators_count=dst_generators_count )
                             ])

            self.last_samples = None

    #override # 返回模型文件名列表
    def get_model_filename_list(self):
        return self.model_filename_list

    #override # 保存模型权重
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def onTrainOneIter(self):
        # 当迭代次数可以被3整除时，更新 warped_src 和 warped_dst
        if self.get_iter() % 3 == 0 and self.last_samples is not None:
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = self.last_samples
            warped_src = target_src
            warped_dst = target_dst
        else:
            samples = self.last_samples = self.generate_next_samples()
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = samples

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm,
                                                 warped_dst, target_dst, target_dstm)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override 
    def onGetPreview(self, samples, for_history=False):
        ( (warped_src, target_src, target_srcm),
          (warped_dst, target_dst, target_dstm) ) = samples

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(4, self.get_batch_size() )
        result = []
        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick224', np.concatenate (st, axis=0 )), ]

        st_m = []
        for i in range(n_samples):
            ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
            st_m.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick224 masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]
        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override # 获取 Merger 的配置信息
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=self.face_type,
                                     default_mode = 'overlay',
                                    )

    def export_dfm(self):
        output_path = self.get_strpath_storage_for_file("model.dfm")

        io.log_info(f"将 .dfm 转储到 {output_path}")

        tf = nn.tf
        nn.set_data_format("NCHW")

        with tf.device(nn.tf_default_device_name):
            warped_dst = tf.placeholder(
                nn.floatx, (None, self.resolution, self.resolution, 3), name="in_face"
            )
            warped_dst = tf.transpose(warped_dst, (0, 3, 1, 2))

            gpu_dst_code = self.inter(self.encoder(warped_dst))
            gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
            _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            gpu_pred_src_dst = tf.transpose(gpu_pred_src_dst, (0, 2, 3, 1))
            gpu_pred_dst_dstm = tf.transpose(gpu_pred_dst_dstm, (0, 2, 3, 1))
            gpu_pred_src_dstm = tf.transpose(gpu_pred_src_dstm, (0, 2, 3, 1))

        tf.identity(gpu_pred_dst_dstm, name="out_face_mask")
        tf.identity(gpu_pred_src_dst, name="out_celeb_face")
        tf.identity(gpu_pred_src_dstm, name="out_celeb_face_mask")

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            nn.tf_sess,
            tf.get_default_graph().as_graph_def(),
            ["out_face_mask", "out_celeb_face", "out_celeb_face_mask"],
        )

        import tf2onnx

        with tf.device("/CPU:0"):
            model_proto, _ = tf2onnx.convert._convert_common(
                output_graph_def,
                name="SAEHD",
                input_names=["in_face:0"],
                output_names=[
                    "out_face_mask:0",
                    "out_celeb_face:0",
                    "out_celeb_face_mask:0",
                ],
                opset=12,
                output_path=output_path,
            )

Model = QModel
