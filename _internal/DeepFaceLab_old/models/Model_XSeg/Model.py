import multiprocessing  # 导入多进程模块
import operator  # 导入运算符模块
from functools import partial  # 从functools模块导入partial，用于固定某些函数参数值

import numpy as np  # 导入numpy并命名为np

from core import mathlib  # 从core包导入mathlib模块
from core.interact import (
    interact as io,
)  # 从core.interact导入interact模块，并重命名为io
from core.leras import nn  # 从core.leras导入nn模块
from facelib import FaceType, XSegNet  # 从facelib导入FaceType和XSegNet模块
from models import ModelBase  # 从models导入ModelBase模块
from samplelib import *  # 从samplelib导入所有


class XSegModel(ModelBase):  # 定义XSegModel类，继承ModelBase

    def __init__(self, *args, **kwargs):  # 类的初始化函数
        super().__init__(*args, **kwargs)  # 调用父类的初始化函数
        # self.resolution = None

    # override
    def on_initialize_options(self):  # 重写on_initialize_options方法
        ask_override = self.ask_override()  # 获取是否覆盖现有选项的标志
        
        min_res = 128
        max_res = 640
        default_resolution = self.options["resolution"] = self.load_or_def_option(
            "resolution", 256
        )
        default_face_type = self.options["face_type"] = self.load_or_def_option(
            "face_type", "wf"
        )  # 加载或设置默认的面部类型选项
        default_pretrain = self.options["pretrain"] = self.load_or_def_option(
            "pretrain", False
        )  # 加载或设置默认的预训练选项
        
        self.options["seg_sample_count"] = self.load_or_def_option(
            "seg_sample_count", 0
        )
        
        if self.is_first_run():  # 如果是第一次运行

            print()

            self.ask_author_name()

            # 获取训练分辨,为64整数倍
            resolution = io.input_int(
                "分辨率 Resolution",
                default_resolution,
                add_info="128-640",
                help_message="更高的分辨率需要更多的 VRAM 和训练时间。该值将调整为64的倍数.",
            )
            resolution = np.clip((resolution // 64) * 64, min_res, max_res)
            self.options["resolution"] = resolution

            self.options["face_type"] = io.input_str(
                "Face type",
                default_face_type,
                ["h", "mf", "f", "wf", "head"],
                help_message="Half / mid face / full face / whole face / head.",
            ).lower()  # 让用户输入面部类型

        if self.is_first_run() or ask_override:  # 如果是第一次运行或需要覆盖选项
            self.ask_batch_size(4, range=[2, 16])  # 设置批处理大小
            self.options["pretrain"] = io.input_bool(
                "启用预训练模式 Enable pretraining mode（请留意预训练文件夹，该模式与正训的算法是不同的）",
                default_pretrain,
            )  # 让用户选择是否启用预训练模式

        if not self.is_exporting and (
            self.options["pretrain"] and self.get_pretraining_data_path() is None
        ):  # 如果未在导出模式且启用了预训练但未设置预训练数据路径
            raise Exception("pretraining_data_path 未定义")  # 抛出异常

        self.pretrain_just_disabled = (
            default_pretrain == True and self.options["pretrain"] == False
        )  # 检查预训练模式是否刚被禁用

    # override
    def on_initialize(self):  # 重写on_initialize方法

        device_config = nn.getCurrentDeviceConfig()  # 获取当前设备配置
        self.model_data_format = (
            "NCHW"
            if self.is_exporting
            or (len(device_config.devices) != 0 and not self.is_debug())
            else "NHWC"
        )  # 设置模型数据格式
        nn.initialize(data_format=self.model_data_format)  # 初始化神经网络库
        tf = nn.tf  # 获取TensorFlow引用

        device_config = nn.getCurrentDeviceConfig()  # 重新获取当前设备配置
        devices = device_config.devices  # 获取设备列表
        self.resolution = resolution = self.options["resolution"]
        self.face_type = {
            "h": FaceType.HALF,
            "mf": FaceType.MID_FULL,
            "f": FaceType.FULL,
            "wf": FaceType.WHOLE_FACE,
            "head": FaceType.HEAD,
        }[self.options["face_type"]]

        place_model_on_cpu = len(devices) == 0  # 如果没有GPU设备，则模型放在CPU上
        models_opt_device = (
            "/CPU:0" if place_model_on_cpu else nn.tf_default_device_name
        )  # 根据是否放置模型在CPU上来选择设备名称

        bgr_shape = nn.get4Dshape(resolution, resolution, 3)  # 获取BGR图像的形状
        mask_shape = nn.get4Dshape(resolution, resolution, 1)  # 获取掩码图像的形状

        # 初始化模型类
        self.model = XSegNet(
            name=self.model_name,
            resolution=resolution,
            load_weights=not self.is_first_run(),  # 如果不是第一次运行，则加载权重
            weights_file_root=self.get_model_root_path(),  # 权重文件的根目录
            training=True,
            place_model_on_cpu=place_model_on_cpu,  # 根据是否放置模型在CPU上
            optimizer=nn.RMSprop(lr=0.0001, lr_dropout=0.3, name="opt"),  # 设置优化器
            data_format=nn.data_format,
        )  # 数据格式

        self.pretrain = self.options["pretrain"]  # 获取预训练选项
        if self.pretrain_just_disabled:  # 如果刚刚禁用了预训练
            self.set_iter(0)  # 重置迭代次数

        if self.is_training:  # 如果在训练模式
            # 根据GPU数量调整批量大小
            gpu_count = max(1, len(devices))  # 计算GPU数量
            bs_per_gpu = max(
                1, self.get_batch_size() // gpu_count
            )  # 计算每个GPU的批量大小
            self.set_batch_size(gpu_count * bs_per_gpu)  # 设置总批量大小

            # 计算每个GPU的损失
            gpu_pred_list = []

            gpu_losses = []
            gpu_loss_gvs = []

            for gpu_id in range(gpu_count):
                with tf.device(
                    f"/{devices[gpu_id].tf_dev_type}:{gpu_id}"
                    if len(devices) != 0
                    else f"/CPU:0"
                ):
                    with tf.device(
                        f"/CPU:0"
                    ):  # 在CPU上分割数据，避免所有数据首先被传输到GPU
                        batch_slice = slice(
                            gpu_id * bs_per_gpu, (gpu_id + 1) * bs_per_gpu
                        )
                        gpu_input_t = self.model.input_t[batch_slice, :, :, :]
                        gpu_target_t = self.model.target_t[batch_slice, :, :, :]

                    # 处理模型张量
                    gpu_pred_logits_t, gpu_pred_t = self.model.flow(
                        gpu_input_t, pretrain=self.pretrain
                    )
                    gpu_pred_list.append(gpu_pred_t)

                    if self.pretrain:  # 如果在预训练模式
                        # 结构损失
                        gpu_loss = tf.reduce_mean(
                            5
                            * nn.dssim(
                                gpu_target_t,
                                gpu_pred_t,
                                max_val=1.0,
                                filter_size=int(resolution / 11.6),
                            ),
                            axis=[1],
                        )
                        gpu_loss += tf.reduce_mean(
                            5
                            * nn.dssim(
                                gpu_target_t,
                                gpu_pred_t,
                                max_val=1.0,
                                filter_size=int(resolution / 23.2),
                            ),
                            axis=[1],
                        )
                        # 像素损失
                        gpu_loss += tf.reduce_mean(
                            10 * tf.square(gpu_target_t - gpu_pred_t), axis=[1, 2, 3]
                        )
                    else:  # 如果不在预训练模式
                        gpu_loss = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=gpu_target_t, logits=gpu_pred_logits_t
                            ),
                            axis=[1, 2, 3],
                        )

                    gpu_losses += [gpu_loss]

                    gpu_loss_gvs += [
                        nn.gradients(gpu_loss, self.model.get_weights())
                    ]  # 计算梯度

            # 计算损失和梯度的平均值，并创建优化器更新操作
            with tf.device(models_opt_device):
                pred = tf.concat(gpu_pred_list, 0)
                loss = tf.concat(gpu_losses, 0)
                loss_gv_op = self.model.opt.get_update_op(
                    nn.average_gv_list(gpu_loss_gvs)
                )

            # 初始化训练和查看函数
            if self.pretrain:

                def train(input_np, target_np):
                    l, _ = nn.tf_sess.run(
                        [loss, loss_gv_op],
                        feed_dict={
                            self.model.input_t: input_np,
                            self.model.target_t: target_np,
                        },
                    )  # 如果是预训练模式，执行训练步骤
                    return l

            else:

                def train(input_np, target_np):
                    l, _ = nn.tf_sess.run(
                        [loss, loss_gv_op],
                        feed_dict={
                            self.model.input_t: input_np,
                            self.model.target_t: target_np,
                        },
                    )  # 如果不是预训练模式，执行训练步骤
                    return l

            self.train = train  # 将训练函数赋值给self.train

            def view(input_np):
                return nn.tf_sess.run(
                    [pred], feed_dict={self.model.input_t: input_np}
                )  # 定义查看函数，用于查看模型输出

            self.view = view

            # 初始化样本生成器
            cpu_count = min(multiprocessing.cpu_count(), 8)  # 获取CPU数量，最多使用8个
            src_dst_generators_count = cpu_count // 2  # 源和目标生成器数量
            src_generators_count = cpu_count // 2  # 源生成器数量
            dst_generators_count = cpu_count // 2  # 目标生成器数量

            if self.pretrain:
                self.options["seg_sample_count"]=0
                pretrain_gen = SampleGeneratorFace(
                    self.get_pretraining_data_path(),
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(random_flip=True),
                    output_sample_types=[
                        {
                            "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                            "warp": True,
                            "transform": True,
                            "channel_type": SampleProcessor.ChannelType.BGR,
                            "face_type": self.face_type,
                            "data_format": nn.data_format,
                            "resolution": resolution,
                        },
                        {
                            "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                            "warp": True,
                            "transform": True,
                            "channel_type": SampleProcessor.ChannelType.G,
                            "face_type": self.face_type,
                            "data_format": nn.data_format,
                            "resolution": resolution,
                        },
                    ],
                    uniform_yaw_distribution=False,
                    generators_count=cpu_count,
                )
                self.set_training_data_generators(
                    [pretrain_gen]
                )  # 如果是预训练模式，设置预训练数据生成器
            else:
                srcdst_generator = SampleGeneratorFaceXSeg(
                    [self.training_data_src_path, self.training_data_dst_path],
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    resolution=resolution,
                    face_type=self.face_type,
                    generators_count=src_dst_generators_count,
                    data_format=nn.data_format,
                )  # 创建源和目标数据的混合生成器
                
                print (srcdst_generator.seg_sample_count)
                self.options["seg_sample_count"]=srcdst_generator.seg_sample_count
                
                src_generator = SampleGeneratorFace(
                    self.training_data_src_path,
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(random_flip=False),
                    output_sample_types=[
                        {
                            "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                            "warp": False,
                            "transform": False,
                            "channel_type": SampleProcessor.ChannelType.BGR,
                            "border_replicate": False,
                            "face_type": self.face_type,
                            "data_format": nn.data_format,
                            "resolution": resolution,
                        }
                    ],
                    generators_count=src_generators_count,
                    raise_on_no_data=False,
                )  # 创建源数据生成器

                dst_generator = SampleGeneratorFace(
                    self.training_data_dst_path,
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(random_flip=False),
                    output_sample_types=[
                        {
                            "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                            "warp": False,
                            "transform": False,
                            "channel_type": SampleProcessor.ChannelType.BGR,
                            "border_replicate": False,
                            "face_type": self.face_type,
                            "data_format": nn.data_format,
                            "resolution": resolution,
                        }
                    ],
                    generators_count=dst_generators_count,
                    raise_on_no_data=False,
                )  # 创建目标数据生成器

                self.set_training_data_generators(
                    [srcdst_generator, src_generator, dst_generator]
                )

    # override
    def get_model_filename_list(self):
        return self.model.model_filename_list  # 返回模型文件名列表

    # override
    def onSave(self):
        self.model.save_weights()  # 保存模型权重

    # override
    def onTrainOneIter(self):
        image_np, target_np = self.generate_next_samples()[0]  # 生成下一批样本
        loss = self.train(image_np, target_np)  # 使用生成的样本进行一次训练迭代

        return (("loss", np.mean(loss)),)  # 返回本次训练迭代的损失

    # override
    def onGetPreview(self, samples, for_history=False):
        n_samples = min(
            4, self.get_batch_size(), 1024 // self.resolution
        )  # 计算预览样本的数量

        if self.pretrain:  # 如果是预训练模式
            (srcdst_samples,) = samples
            image_np, mask_np = srcdst_samples  # 获取图像和掩码样本
        else:
            srcdst_samples, src_samples, dst_samples = samples
            image_np, mask_np = srcdst_samples  # 获取源和目标样本

        (
            I,
            M,
            IM,
        ) = [
            np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
            for x in ([image_np, mask_np] + self.view(image_np))
        ]  # 转换数据格式并裁剪值范围
        (
            M,
            IM,
        ) = [
            np.repeat(x, (3,), -1) for x in [M, IM]
        ]  # 将掩码重复三次以适应颜色通道

        green_bg = np.tile(
            np.array([0, 1, 0], dtype=np.float32)[None, None, ...],
            (self.resolution, self.resolution, 1),
        )  # 创建绿色背景

        result = []
        st = []
        for i in range(n_samples):
            if self.pretrain:
                ar = I[i], IM[i]  # 如果是预训练模式，只显示原图和掩码图
            else:
                ar = (
                    I[i] * M[i] + 0.5 * I[i] * (1 - M[i]) + 0.5 * green_bg * (1 - M[i]),
                    IM[i],
                    I[i] * IM[i]
                    + 0.5 * I[i] * (1 - IM[i])
                    + 0.5 * green_bg * (1 - IM[i]),
                )
            st.append(np.concatenate(ar, axis=1))
        result += [
            ("XSeg training faces", np.concatenate(st, axis=0)),
        ]

        if (
            not self.pretrain and len(src_samples) != 0
        ):  # 如果不是预训练模式且源样本不为空
            (src_np,) = src_samples

            (
                D,
                DM,
            ) = [
                np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
                for x in ([src_np] + self.view(src_np))
            ]
            (DM,) = [np.repeat(x, (3,), -1) for x in [DM]]

            st = []
            for i in range(n_samples):
                ar = (
                    D[i],
                    DM[i],
                    D[i] * DM[i]
                    + 0.5 * D[i] * (1 - DM[i])
                    + 0.5 * green_bg * (1 - DM[i]),
                )
                st.append(np.concatenate(ar, axis=1))

            result += [
                ("XSeg src faces", np.concatenate(st, axis=0)),
            ]

        if (
            not self.pretrain and len(dst_samples) != 0
        ):  # 如果不是预训练模式且目标样本不为空
            (dst_np,) = dst_samples

            (
                D,
                DM,
            ) = [
                np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
                for x in ([dst_np] + self.view(dst_np))
            ]
            (DM,) = [np.repeat(x, (3,), -1) for x in [DM]]

            st = []
            for i in range(n_samples):
                ar = (
                    D[i],
                    DM[i],
                    D[i] * DM[i]
                    + 0.5 * D[i] * (1 - DM[i])
                    + 0.5 * green_bg * (1 - DM[i]),
                )
                st.append(np.concatenate(ar, axis=1))

            result += [
                ("XSeg dst faces", np.concatenate(st, axis=0)),
            ]

        return result  # 返回预览结果列表

    def export_dfm(self):
        io.log_info(f"XSeg模型不可导出.dfm")  # 日志信息显示导出路径


Model = XSegModel