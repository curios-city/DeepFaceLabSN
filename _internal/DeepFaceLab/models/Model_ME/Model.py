import multiprocessing
import operator

import numpy as np

import os
import shutil

# from psutil import cpu_count

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

from pathlib import Path

from utils.label_face import label_face_filename

from utils.train_status_export import data_format_change, prepare_sample
import cv2
from core.cv2ex import cv2_imwrite
from tqdm import tqdm


class MEModel(ModelBase):
    # 重写父类的on_initialize_options方法
    def on_initialize_options(self):
        # 获取当前的设备配置
        device_config = nn.getCurrentDeviceConfig()

        # 根据设备的VRAM容量建议批处理大小
        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb
        if lowest_vram >= 4:
            suggest_batch_size = 8
        else:
            suggest_batch_size = 4

        # 定义最小和最大分辨率
        min_res = 64
        max_res = 640
        default_usefp16 = self.options["use_fp16"] = self.load_or_def_option(
            "use_fp16", False
        )
        default_resolution = self.options["resolution"] = self.load_or_def_option(
            "resolution", 128
        )
        default_face_type = self.options["face_type"] = self.load_or_def_option(
            "face_type", "f"
        )
        default_models_opt_on_gpu = self.options["models_opt_on_gpu"] = (
            self.load_or_def_option("models_opt_on_gpu", True)
        )

        default_archi = self.options["archi"] = self.load_or_def_option(
            "archi", "liae-ud"
        )

        default_ae_dims = self.options["ae_dims"] = self.load_or_def_option(
            "ae_dims", 256
        )
        default_e_dims = self.options["e_dims"] = self.load_or_def_option("e_dims", 64)
        default_d_dims = self.options["d_dims"] = self.options.get("d_dims", None)
        default_d_mask_dims = self.options["d_mask_dims"] = self.options.get(
            "d_mask_dims", None
        )
        default_masked_training = self.options["masked_training"] = (
            self.load_or_def_option("masked_training", True)
        )

        default_retraining_samples = self.options["retraining_samples"] = (
            self.load_or_def_option("retraining_samples", False)
        )

        default_eyes_prio = self.options["eyes_prio"] = self.load_or_def_option(
            "eyes_prio", False
        )
        default_mouth_prio = self.options["mouth_prio"] = self.load_or_def_option(
            "mouth_prio", False
        )

        # Compatibility check
        eyes_mouth_prio = self.options.get("eyes_mouth_prio")
        if eyes_mouth_prio is not None:
            default_eyes_prio = self.options["eyes_prio"] = eyes_mouth_prio
            default_mouth_prio = self.options["mouth_prio"] = eyes_mouth_prio
            self.options.pop("eyes_mouth_prio")

        default_uniform_yaw = self.options["uniform_yaw"] = self.load_or_def_option(
            "uniform_yaw", False
        )
        default_blur_out_mask = self.options["blur_out_mask"] = self.load_or_def_option(
            "blur_out_mask", False
        )

        default_adabelief = self.options["adabelief"] = self.load_or_def_option(
            "adabelief", True
        )

        lr_dropout = self.load_or_def_option("lr_dropout", "n")
        lr_dropout = {True: "y", False: "n"}.get(
            lr_dropout, lr_dropout
        )  # backward comp
        default_lr_dropout = self.options["lr_dropout"] = lr_dropout

        default_loss_function = self.options["loss_function"] = self.load_or_def_option(
            "loss_function", "SSIM"
        )

        default_random_warp = self.options["random_warp"] = self.load_or_def_option(
            "random_warp", True
        )
        default_random_hsv_power = self.options["random_hsv_power"] = (
            self.load_or_def_option("random_hsv_power", 0.0)
        )
        default_random_downsample = self.options["random_downsample"] = (
            self.load_or_def_option("random_downsample", False)
        )
        default_random_noise = self.options["random_noise"] = self.load_or_def_option(
            "random_noise", False
        )
        default_random_blur = self.options["random_blur"] = self.load_or_def_option(
            "random_blur", False
        )
        default_random_jpeg = self.options["random_jpeg"] = self.load_or_def_option(
            "random_jpeg", False
        )
        default_super_warp = self.options["super_warp"] = self.load_or_def_option(
            "super_warp", False
        )
        default_rotation_range = self.rotation_range = [-3, 3]
        default_scale_range = self.scale_range = [-0.15, 0.15]

        # 加载或定义其他训练相关的默认选项
        default_background_power = self.options["background_power"] = (
            self.load_or_def_option("background_power", 0.0)
        )
        default_true_face_power = self.options["true_face_power"] = (
            self.load_or_def_option("true_face_power", 0.0)
        )
        default_face_style_power = self.options["face_style_power"] = (
            self.load_or_def_option("face_style_power", 0.0)
        )
        default_bg_style_power = self.options["bg_style_power"] = (
            self.load_or_def_option("bg_style_power", 0.0)
        )
        default_ct_mode = self.options["ct_mode"] = self.load_or_def_option(
            "ct_mode", "none"
        )
        default_random_color = self.options["random_color"] = self.load_or_def_option(
            "random_color", False
        )
        default_clipgrad = self.options["clipgrad"] = self.load_or_def_option(
            "clipgrad", False
        )
        default_pretrain = self.options["pretrain"] = self.load_or_def_option(
            "pretrain", False
        )
        default_cpu_cap = self.options["cpu_cap"] = self.load_or_def_option(
            "cpu_cap", 8
        )
        default_preview_samples = self.options["preview_samples"] = (
            self.load_or_def_option("preview_samples", 4)
        )
        default_full_preview = self.options["force_full_preview"] = (
            self.load_or_def_option("force_full_preview", False)
        )
        default_lr = self.options["lr"] = self.load_or_def_option("lr", 5e-5)

        # 判断是否需要覆盖模型设置
        ask_override = False if self.read_from_conf else self.ask_override()
        if self.is_first_run() or ask_override:
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                # 如果是首次运行或需要覆盖设置，则询问用户输入各种配置
                self.ask_autobackup_hour()
                self.ask_maximum_n_backups()
                self.ask_write_preview_history()
                self.options["preview_samples"] = np.clip(
                    io.input_int(
                        "预览样本数量（纵向）",
                        default_preview_samples,
                        add_info="1 - 6",
                        help_message="典型的精细值为4",
                    ),
                    1,
                    16,
                )
                self.options["force_full_preview"] = io.input_bool(
                    "强制不分离预览", default_full_preview,
                    help_message="遇到大分辨率也会展开五列",
                )

                # 获取其他训练相关配置
                self.ask_reset_training()
                self.ask_target_iter()
                self.ask_retraining_samples(default_retraining_samples)
                self.ask_random_src_flip()
                self.ask_random_dst_flip()
                self.ask_batch_size(suggest_batch_size)
                self.options["use_fp16"] = io.input_bool(
                    "使用fp16（测试功能）",
                    default_usefp16,
                    help_message="提高训练速度，减少显存占用，可增加BS上限。前期易崩溃，后期精度不够，建议5000~200000迭代使用, 务必先备份！",
                )
                self.options["cpu_cap"] = np.clip(
                    io.input_int(
                        "最大使用的 CPU 核心数.",
                        default_cpu_cap,
                        add_info="1 - 256",
                        help_message="典型的精细值为 8",
                    ),
                    1,
                    256,
                )

        if self.is_first_run():
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                # 获取训练分辨
                resolution = io.input_int(
                    "分辨率 Resolution",
                    default_resolution,
                    add_info="64-640",
                    help_message="更高的分辨率需要更多的 VRAM 和训练时间。该值将调整为 16 和 32 的倍数，以适应不同的架构.",
                )
                resolution = np.clip((resolution // 16) * 16, min_res, max_res)
                self.options["resolution"] = resolution
                self.options["face_type"] = io.input_str(
                    "人脸类型 Face_type",
                    default_face_type,
                    ["h", "mf", "f", "wf", "head", "custom"],
                    help_message="Half / mid face / full face / whole face / head / custom. 半脸/中脸/全脸/全脸/头部/自定义。半脸的分辨率较高，但覆盖脸颊的面积较小。中脸比半脸宽 30%。全脸 包括前额在内的整个脸部。头部覆盖整个头部，但需要 XSeg 来获取源和目的面部集",
                ).lower()

                # 获取训练架构配置
                while True:
                    archi = io.input_str(
                        "AE architecture",
                        default_archi,
                        help_message="""
                            'df' 保持更多身份特征的脸部（更像SRC）。
                            'liae' 可以修复过于不同的脸部形状（更像DST）。
                            '-u' 增加与源人脸（SRC）的相似度，需要更多 VRAM。
                            '-d' 计算成本减半。需要更长的训练时间，并建议使用预训练模型。分辨率必须按 32 的倍数更改
							'-t' 增加与源人脸（SRC）的相似度。
							'-c' （实验性）将激活函数设置为余弦单位（默认值：Leaky ReLu）。
                            示例: df, liae-d, df-dt, liae-udt, ...
                            """,
                    ).lower()

                    archi_split = archi.split("-")

                    if len(archi_split) == 2:
                        archi_type, archi_opts = archi_split
                    elif len(archi_split) == 1:
                        archi_type, archi_opts = archi_split[0], None
                    else:
                        continue

                    if archi_type not in ["df", "liae"]:
                        continue

                    if archi_opts is not None:
                        if len(archi_opts) == 0:
                            continue
                        if (
                            len(
                                [
                                    1
                                    for opt in archi_opts
                                    if opt not in ["u", "d", "t", "c"]
                                ]
                            )
                            != 0
                        ):
                            continue

                        if "d" in archi_opts:
                            self.options["resolution"] = np.clip(
                                (self.options["resolution"] // 32) * 32,
                                min_res,
                                max_res,
                            )

                    break
                self.options["archi"] = archi

            default_d_dims = self.options["d_dims"] = self.load_or_def_option(
                "d_dims", 64
            )

            default_d_mask_dims = default_d_dims // 3
            default_d_mask_dims += default_d_mask_dims % 2
            default_d_mask_dims = self.options["d_mask_dims"] = self.load_or_def_option(
                "d_mask_dims", default_d_mask_dims
            )
            # 作者签名
            self.ask_author_name()

        # 首次运行时获取AutoEncoder、Encoder和Decoder的维度配置
        if self.is_first_run():
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                self.options["ae_dims"] = np.clip(
                    io.input_int(
                        "自动编码器尺寸 AutoEncoder dimensions",
                        default_ae_dims,
                        add_info="32-1024",
                        help_message="所有脸部信息将被压缩到AE维度。如果AE维度不够，例如闭上的眼睛可能无法识别。更多的维度意味着更好，但需要更多VRAM。可以根据GPU调整模型大小。",
                    ),
                    32,
                    1024,
                )

                e_dims = np.clip(
                    io.input_int(
                        "编码器尺寸 Encoder dimensions",
                        default_e_dims,
                        add_info="16-256",
                        help_message="更多维度有助于识别更多面部特征并获得更清晰的结果，但需要更多VRAM。可以根据GPU调整模型大小。",
                    ),
                    16,
                    256,
                )
                self.options["e_dims"] = e_dims + e_dims % 2

                d_dims = np.clip(
                    io.input_int(
                        "解码器尺寸 Decoder dimensions",
                        default_d_dims,
                        add_info="16-256",
                        help_message="更多维度有助于识别更多面部特征并获得更清晰的结果，但需要更多VRAM。可以根据GPU调整模型大小。",
                    ),
                    16,
                    256,
                )
                self.options["d_dims"] = d_dims + d_dims % 2

                d_mask_dims = np.clip(
                    io.input_int(
                        "解码器遮罩尺寸 Decoder mask dimensions",
                        default_d_mask_dims,
                        add_info="16-256",
                        help_message="典型的遮罩维度是解码器维度的三分之一。如果你手动从目标遮罩中剔除障碍物，可以增加此参数以获得更好的质量。",
                    ),
                    16,
                    256,
                )

                self.options["adabelief"] = io.input_bool(
                    "使用AdaBelief优化器 Use AdaBelief optimizer?",
                    default_adabelief,
                    help_message="使用 AdaBelief 优化器。它需要更多的 VRAM，但模型的准确性和泛化程度更高",
                )

                self.options["d_mask_dims"] = d_mask_dims + d_mask_dims % 2

        # 首次运行或需要覆盖设置时的配置
        if self.is_first_run() or ask_override:
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                # 特定脸部类型的额外配置
                if self.options["face_type"] in ["wf", "head", "custom"]:
                    self.options["masked_training"] = io.input_bool(
                        "遮罩训练 Masked training",
                        default_masked_training,
                        help_message="此选项仅适用于'whole_face'或'head'类型。遮罩训练将训练区域剪辑到全脸遮罩或XSeg遮罩，从而更好地训练脸部。",
                    )

                # 获取眼睛和嘴巴优先级配置
                self.options["eyes_prio"] = io.input_bool(
                    "眼睛优先 Eyes priority",
                    default_eyes_prio,
                    help_message='有助于解决训练中的眼睛问题，如"异形眼"和错误的眼睛方向（特别是在高分辨率训练），通过强制神经网络优先训练眼睛。',
                )
                self.options["mouth_prio"] = io.input_bool(
                    "嘴巴优先 Mouth priority",
                    default_mouth_prio,
                    help_message="有助于通过强制神经网络优先训练嘴巴来解决训练中的嘴巴问题。",
                )

                # 获取其他训练配置
                self.options["uniform_yaw"] = io.input_bool(
                    "侧脸优化 Uniform yaw distribution of samples",
                    default_uniform_yaw,
                    help_message="有助于解决样本中侧脸模糊的问题，由于侧脸在数据集中数量较少。",
                )
                self.options["blur_out_mask"] = io.input_bool(
                    "遮罩边缘模糊 Blur out mask",
                    default_blur_out_mask,
                    help_message="在训练样本的应用面部遮罩的外围区域进行模糊处理。结果是脸部附近的背景平滑且在换脸时不那么明显。需要在源和目标数据集中使用精确的xseg遮罩。",
                )

        # GAN相关配置
        default_gan_power = self.options["gan_power"] = self.load_or_def_option(
            "gan_power", 0.0
        )
        default_gan_patch_size = self.options["gan_patch_size"] = (
            self.load_or_def_option("gan_patch_size", self.options["resolution"] // 8)
        )
        default_gan_dims = self.options["gan_dims"] = self.load_or_def_option(
            "gan_dims", 16
        )
        default_gan_smoothing = self.options["gan_smoothing"] = self.load_or_def_option(
            "gan_smoothing", 0.1
        )
        default_gan_noise = self.options["gan_noise"] = self.load_or_def_option(
            "gan_noise", 0.0
        )

        if self.is_first_run() or ask_override:
            if (
                self.read_from_conf and not self.config_file_exists
            ) or not self.read_from_conf:
                self.options["models_opt_on_gpu"] = io.input_bool(
                    "将模型和优化器放到GPU上 Place models and optimizer on GPU",
                    default_models_opt_on_gpu,
                    help_message="在一个 GPU 上进行训练时，默认情况下模型和优化器权重会放在 GPU 上，以加速训练过程。您可以将它们放在 CPU 上，以释放额外的 VRAM，从而设置更大的维度",
                )

                self.options["lr_dropout"] = io.input_str(
                    f"使用学习率下降 Use learning rate dropout",
                    default_lr_dropout,
                    ["n", "y", "cpu"],
                    help_message="当人脸训练得足够好时，可以启用该选项来获得额外的清晰度，并减少子像素抖动，从而减少迭代次数。在 禁用随机扭曲 和 GAN 之前启用。在 CPU 上启用。这样就可以不使用额外的 VRAM，牺牲 20% 的迭代时间",
                )

                self.options["loss_function"] = io.input_str(
                    f"损失函数 Loss function",
                    default_loss_function,
                    ["SSIM", "MS-SSIM", "MS-SSIM+L1"],
                    help_message="用于图像质量评估的变化损失函数",
                )

                self.options["lr"] = np.clip(
                    io.input_number(
                        "学习率 Learning rate",
                        default_lr,
                        add_info="0.0 .. 1.0",
                        help_message="学习率：典型精细值 5e-5",
                    ),
                    0.0,
                    1,
                )

                self.options["random_warp"] = io.input_bool(
                    "启用样本随机扭曲 Enable random warp of samples",
                    default_random_warp,
                    help_message="要概括两张人脸的面部表情，需要使用随机翘曲。当人脸训练得足够好时，可以禁用它来获得额外的清晰度，并减少亚像素抖动，从而减少迭代次数",
                )

                self.options["random_hsv_power"] = np.clip(
                    io.input_number(
                        "随机色调/饱和度/光强度 Random hue/saturation/light intensity",
                        default_random_hsv_power,
                        add_info="0.0 .. 0.3",
                        help_message="随机色调/饱和度/光照强度仅应用于神经网络输入的src人脸集。稳定人脸交换过程中的色彩扰动。通过选择原始面孔集中最接近的面孔来降低色彩转换的质量。因此src人脸集必须足够多样化。典型的精细值为 0.05",
                    ),
                    0.0,
                    0.3,
                )

                self.options["random_downsample"] = io.input_bool(
                    "启用样本随机降低采样 Enable random downsample of samples",
                    default_random_downsample,
                    help_message="通过缩小部分样本来挑战模型",
                )
                self.options["random_noise"] = io.input_bool(
                    "启用在样本中随机添加噪音 Enable random noise added to samples",
                    default_random_noise,
                    help_message="通过在某些样本中添加噪音来挑战模型",
                )
                self.options["random_blur"] = io.input_bool(
                    "启用对样本的随机模糊 Enable random blur of samples",
                    default_random_blur,
                    help_message="通过在某些样本中添加模糊效果来挑战模型",
                )
                self.options["random_jpeg"] = io.input_bool(
                    "启用随机压缩jpeg样本 Enable random jpeg compression of samples",
                    default_random_jpeg,
                    help_message="通过对某些样本应用 jpeg 压缩的质量降级来挑战模型",
                )

                self.options["super_warp"] = io.input_bool(
                    "启用样本超级扭曲 Enable super warp of samples",
                    default_super_warp,
                    help_message="大多数时候不要开启，占用更多时间和空间。只有dst有夸张大幅度表情，而src无对应表情时，可以尝试增大计算量以求融合。或许搭配嘴巴优先 Mouth priority更有针对性！",
                )

                # if self.options["super_warp"] == True:
                # self.rotation_range=[-15,15]
                # self.scale_range=[-0.25, 0.25]

                """
                self.options["random_shadow"] = io.input_str(
                    "启用对样本的随机阴影和高光 Enable random shadows and highlights of samples",
                    default_random_shadow,
                    ["none", "src", "dst", "all"],
                    help_message="有助于在数据集中创建暗光区域。如果你的src数据集缺乏阴影/不同的光照情况；使用dst以帮助泛化；或者使用all以满足两者的需求",
                )
                """
                self.options["gan_power"] = np.clip(
                    io.input_number(
                        "GAN强度 GAN power",
                        default_gan_power,
                        add_info="0.0 .. 10.0",
                        help_message="以生成对抗方式训练网络。强制神经网络学习人脸的小细节。只有当人脸训练得足够好时才启用它，否则就不要禁用。典型值为 0.1",
                    ),
                    0.0,
                    10.0,
                )

                if self.options["gan_power"] != 0.0:
                    gan_patch_size = np.clip(
                        io.input_int(
                            "GAN补丁大小 GAN patch size",
                            default_gan_patch_size,
                            add_info="3-640",
                            help_message="补丁大小越大，质量越高，需要的显存越多。即使在最低设置下，您也可以获得更清晰的边缘。典型的良好数值是分辨率除以8",
                        ),
                        3,
                        640,
                    )
                    self.options["gan_patch_size"] = gan_patch_size

                    gan_dims = np.clip(
                        io.input_int(
                            "GAN维度 GAN dimensions",
                            default_gan_dims,
                            add_info="4-64",
                            help_message="GAN 网络的尺寸。尺寸越大，所需的 VRAM 越多。即使在最低设置下，也能获得更清晰的边缘。典型的精细值为 16",
                        ),
                        4,
                        64,
                    )
                    self.options["gan_dims"] = gan_dims

                    self.options["gan_smoothing"] = np.clip(
                        io.input_number(
                            "GAN标签平滑 GAN label smoothing",
                            default_gan_smoothing,
                            add_info="0 - 0.5",
                            help_message="使用软标签，其值略微偏离 GAN 的 0/1，具有正则化效果",
                        ),
                        0,
                        0.5,
                    )
                    self.options["gan_noise"] = np.clip(
                        io.input_number(
                            "GAN噪声标签 GAN noisy labels",
                            default_gan_noise,
                            add_info="0 - 0.5",
                            help_message="用错误的标签标记某些图像，有助于防止塌陷",
                        ),
                        0,
                        0.5,
                    )

                if "df" in self.options["archi"]:
                    self.options["true_face_power"] = np.clip(
                        io.input_number(
                            "真脸(src)强度 'True face' power.",
                            default_true_face_power,
                            add_info="0.0000 .. 1.0",
                            help_message="实验选项。判别结果面孔更像原始面孔。数值越大，判别能力越强。典型值为0.01。比较 - https://i.imgur.com/czScS9q.png",
                        ),
                        0.0,
                        1.0,
                    )
                else:
                    self.options["true_face_power"] = 0.0

                self.options["background_power"] = np.clip(
                    io.input_number(
                        "背景(src)强度 Background power",
                        default_background_power,
                        add_info="0.0..1.0",
                        help_message="了解遮罩外的区域。有助于平滑遮罩边界附近的区域。可随时使用",
                    ),
                    0.0,
                    1.0,
                )

                self.options["face_style_power"] = np.clip(
                    io.input_number(
                        "人脸(dst)强度 Face style power",
                        default_face_style_power,
                        add_info="0.0..100.0",
                        help_message="学习预测脸部的颜色，使其与遮罩内的 dst 相同。如果要将此选项与whole_face一起使用，则必须使用 XSeg 训练掩码。警告： 只有在 10k 次之后，当预测的面部足够清晰，可以开始学习风格时，才能启用该选项。从 0.001 值开始，检查历史变化。启用此选项会增加模型崩溃的几率",
                    ),
                    0.0,
                    100.0,
                )
                self.options["bg_style_power"] = np.clip(
                    io.input_number(
                        "背景(dst)强度 Background style power",
                        default_bg_style_power,
                        add_info="0.0..100.0",
                        help_message="学习预测脸部遮罩外的区域与 dst 相同。如果要将此选项用于whole_face，则必须使用 XSeg 训练蒙板。对于 whole_face，你必须使用 XSeg 训练蒙板。这会使脸部更像 dst。启用此选项会增加模型崩溃的几率。典型值为 2.0",
                    ),
                    0.0,
                    100.0,
                )

                self.options["ct_mode"] = io.input_str(
                    f"色彩转换模式 Color transfer for src faceset",
                    default_ct_mode,
                    ["none", "rct", "lct", "mkl", "idt", "sot", "fs-aug", "cc-aug"],
                    help_message="改变靠近 dst 样本的 src 样本的颜色分布。尝试所有模式，找出最佳方案。CC和FS aug 为 dst 和 src 添加随机颜色",
                )
                self.options["random_color"] = io.input_bool(
                    "随机颜色 Random color",
                    default_random_color,
                    help_message="在LAB色彩空间中，样本随机围绕 L 轴旋转，有助于训练泛化。说人话就是亮度不变，但是色相变化比hsv大。hsv的亮度和对比确实不建议大幅度，所以本选项是互补的，建议是轮流开启而非同时开启！",
                )
                self.options["clipgrad"] = io.input_bool(
                    "启用梯度裁剪 Enable gradient clipping",
                    default_clipgrad,
                    help_message="梯度裁剪降低了模型崩溃的几率，但牺牲了训练速度",
                )

                self.options["pretrain"] = io.input_bool(
                    "启用预训练模式 Enable pretraining mode",
                    default_pretrain,
                    help_message="使用大量各种人脸预训练模型,模型可用于更快地训练伪造数据.强制使用 random_warp=N, random_flips=Y, gan_power=0.0, lr_dropout=N, styles=0.0, uniform_yaw=Y",
                )

        if self.options["pretrain"] and self.get_pretraining_data_path() is None:
            raise Exception("pretraining_data_path is not defined")

        self.gan_model_changed = (
            default_gan_patch_size != self.options["gan_patch_size"]
        ) or (default_gan_dims != self.options["gan_dims"])
        # 预训转正
        self.pretrain_just_disabled = (
            default_pretrain == True and self.options["pretrain"] == False
        )

    # 重写父类的on_initialize方法
    def on_initialize(self):
        # 获取当前设备配置和初始化数据格式
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = (
            "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        )
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf  # TensorFlow引用

        # 设置分辨率和脸部类型
        self.resolution = resolution = self.options["resolution"]
        self.face_type = {
            "h": FaceType.HALF,
            "mf": FaceType.MID_FULL,
            "f": FaceType.FULL,
            "wf": FaceType.WHOLE_FACE,
            "custom": FaceType.CUSTOM,
            "head": FaceType.HEAD,
        }[self.options["face_type"]]

        # 设置眼睛和嘴巴优先级
        eyes_prio = self.options["eyes_prio"]
        mouth_prio = self.options["mouth_prio"]

        # 解析架构类型
        archi_split = self.options["archi"].split("-")
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None
        self.archi_type = archi_type

        # 设置AutoEncoder、Encoder和Decoder的维度
        ae_dims = self.options["ae_dims"]
        e_dims = self.options["e_dims"]
        d_dims = self.options["d_dims"]
        d_mask_dims = self.options["d_mask_dims"]

        # 设置是否预训练
        self.pretrain = self.options["pretrain"]
        if self.pretrain_just_disabled:
            ask_for_clean = input("是否将迭代数归零？请输入 'y' 或 'n': ")
            if ask_for_clean.lower() == "y":
                self.set_iter(0)
                print("迭代数已重置！")
            else:
                print("保留迭代数结束预训！")

        # 设置是否使用AdaBelief优化器
        adabelief = self.options["adabelief"]

        # 设置是否使用半精度浮点数
        use_fp16 = self.options["use_fp16"]
        if self.is_exporting:
            use_fp16 = io.input_bool(
                "Export quantized?",
                False,
                help_message="使导出的模型更快。如果遇到问题，请禁用此选项。",
            )

        # 设置相关参数 （已解锁预训练的所有锁定，除了GAN）
        self.gan_power = gan_power = 0.0 if self.pretrain else self.options["gan_power"]
        random_warp = self.options["random_warp"]
        random_src_flip = self.random_src_flip
        random_dst_flip = self.random_dst_flip
        random_hsv_power = self.options["random_hsv_power"]
        blur_out_mask = self.options["blur_out_mask"]

        # 如果处于预训练阶段，调整一些参数设置（已解RW\flip\hsv\blur,保留gan和style的限制）
        if self.pretrain:
            self.options_show_override["gan_power"] = 0.0
            self.options_show_override["face_style_power"] = 0.0
            self.options_show_override["bg_style_power"] = 0.0

        # 设置是否进行遮罩训练和颜色转换模式
        masked_training = self.options["masked_training"]
        ct_mode = self.options["ct_mode"]
        if ct_mode == "none":
            ct_mode = None

        """
        # 根据配置文件的使用情况设置随机阴影源和目标
        if (
            self.read_from_conf and not self.config_file_exists
        ) or not self.read_from_conf:
            random_shadow_src = (
                True if self.options["random_shadow"] in ["all", "src"] else False
            )
            random_shadow_dst = (
                True if self.options["random_shadow"] in ["all", "dst"] else False
            )

            # 如果是首次使用配置文件创建模型，则删除随机阴影选项
            if not self.config_file_exists and self.read_from_conf:
                del self.options["random_shadow"]
        else:
            random_shadow_src = self.options["random_shadow_src"]
            random_shadow_dst = self.options["random_shadow_dst"]
        """

        # 设置模型优化选项
        models_opt_on_gpu = (
            False if len(devices) == 0 else self.options["models_opt_on_gpu"]
        )
        models_opt_device = (
            nn.tf_default_device_name
            if models_opt_on_gpu and self.is_training
            else "/CPU:0"
        )
        optimizer_vars_on_cpu = models_opt_device == "/CPU:0"

        # 设置输入通道和形状
        input_ch = 3
        bgr_shape = self.bgr_shape = nn.get4Dshape(resolution, resolution, input_ch)
        mask_shape = nn.get4Dshape(resolution, resolution, 1)
        self.model_filename_list = []

        with tf.device("/CPU:0"):
            # 在CPU上初始化占位符
            self.warped_src = tf.placeholder(nn.floatx, bgr_shape, name="warped_src")
            self.warped_dst = tf.placeholder(nn.floatx, bgr_shape, name="warped_dst")

            self.target_src = tf.placeholder(nn.floatx, bgr_shape, name="target_src")
            self.target_dst = tf.placeholder(nn.floatx, bgr_shape, name="target_dst")

            self.target_srcm = tf.placeholder(nn.floatx, mask_shape, name="target_srcm")
            self.target_srcm_em = tf.placeholder(
                nn.floatx, mask_shape, name="target_srcm_em"
            )
            self.target_dstm = tf.placeholder(nn.floatx, mask_shape, name="target_dstm")
            self.target_dstm_em = tf.placeholder(
                nn.floatx, mask_shape, name="target_dstm_em"
            )

        # 初始化模型架构
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=use_fp16, opts=archi_opts)

        # 继续初始化模型的其他组件
        with tf.device(models_opt_device):
            # 根据架构类型初始化模型的不同部分
            if "df" in archi_type:
                # DF架构
                self.encoder = model_archi.Encoder(
                    in_ch=input_ch, e_ch=e_dims, name="encoder"
                )
                encoder_out_ch = (
                    self.encoder.get_out_ch()
                    * self.encoder.get_out_res(resolution) ** 2
                )

                self.inter = model_archi.Inter(
                    in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name="inter"
                )
                inter_out_ch = self.inter.get_out_ch()

                self.decoder_src = model_archi.Decoder(
                    in_ch=inter_out_ch,
                    d_ch=d_dims,
                    d_mask_ch=d_mask_dims,
                    name="decoder_src",
                )
                self.decoder_dst = model_archi.Decoder(
                    in_ch=inter_out_ch,
                    d_ch=d_dims,
                    d_mask_ch=d_mask_dims,
                    name="decoder_dst",
                )

                self.model_filename_list += [
                    [self.encoder, "encoder.npy"],
                    [self.inter, "inter.npy"],
                    [self.decoder_src, "decoder_src.npy"],
                    [self.decoder_dst, "decoder_dst.npy"],
                ]

                # 如果正在训练，初始化代码鉴别器
                if self.is_training:
                    if self.options["true_face_power"] != 0:
                        self.code_discriminator = nn.CodeDiscriminator(
                            ae_dims, code_res=self.inter.get_out_res(), name="dis"
                        )
                        self.model_filename_list += [
                            [self.code_discriminator, "code_discriminator.npy"]
                        ]

            elif "liae" in archi_type:
                # LIAE架构
                self.encoder = model_archi.Encoder(
                    in_ch=input_ch, e_ch=e_dims, name="encoder"
                )
                encoder_out_ch = (
                    self.encoder.get_out_ch()
                    * self.encoder.get_out_res(resolution) ** 2
                )

                self.inter_AB = model_archi.Inter(
                    in_ch=encoder_out_ch,
                    ae_ch=ae_dims,
                    ae_out_ch=ae_dims * 2,
                    name="inter_AB",
                )
                self.inter_B = model_archi.Inter(
                    in_ch=encoder_out_ch,
                    ae_ch=ae_dims,
                    ae_out_ch=ae_dims * 2,
                    name="inter_B",
                )

                inter_out_ch = self.inter_AB.get_out_ch()
                inters_out_ch = inter_out_ch * 2
                self.decoder = model_archi.Decoder(
                    in_ch=inters_out_ch,
                    d_ch=d_dims,
                    d_mask_ch=d_mask_dims,
                    name="decoder",
                )

                self.model_filename_list += [
                    [self.encoder, "encoder.npy"],
                    [self.inter_AB, "inter_AB.npy"],
                    [self.inter_B, "inter_B.npy"],
                    [self.decoder, "decoder.npy"],
                ]

            if self.is_training:
                if gan_power != 0:
                    self.D_src = nn.UNetPatchDiscriminator(
                        patch_size=self.options["gan_patch_size"],
                        in_ch=input_ch,
                        base_ch=self.options["gan_dims"],
                        use_fp16=self.options["use_fp16"],
                        name="D_src",
                    )
                    self.model_filename_list += [[self.D_src, "GAN.npy"]]

                # 初始化优化器
                lr = self.options["lr"]

                if self.options["lr_dropout"] in ["y", "cpu"] and not self.pretrain:
                    lr_cos = 500
                    lr_dropout = 0.3
                else:
                    lr_cos = 0
                    lr_dropout = 1.0
                OptimizerClass = nn.AdaBelief if adabelief else nn.RMSprop
                clipnorm = 1.0 if self.options["clipgrad"] else 0.0

                # 根据架构类型设置优化器
                if "df" in archi_type:
                    self.src_dst_saveable_weights = (
                        self.encoder.get_weights()
                        + self.inter.get_weights()
                        + self.decoder_src.get_weights()
                        + self.decoder_dst.get_weights()
                    )
                    self.src_dst_trainable_weights = self.src_dst_saveable_weights
                elif "liae" in archi_type:
                    self.src_dst_saveable_weights = (
                        self.encoder.get_weights()
                        + self.inter_AB.get_weights()
                        + self.inter_B.get_weights()
                        + self.decoder.get_weights()
                    )
                    if random_warp:
                        self.src_dst_trainable_weights = self.src_dst_saveable_weights
                    else:
                        self.src_dst_trainable_weights = (
                            self.encoder.get_weights()
                            + self.inter_B.get_weights()
                            + self.decoder.get_weights()
                        )

                # 初始化源和目标的优化器
                self.src_dst_opt = OptimizerClass(
                    lr=lr,
                    lr_dropout=lr_dropout,
                    lr_cos=lr_cos,
                    clipnorm=clipnorm,
                    name="src_dst_opt",
                )
                self.src_dst_opt.initialize_variables(
                    self.src_dst_saveable_weights,
                    vars_on_cpu=optimizer_vars_on_cpu,
                    lr_dropout_on_cpu=self.options["lr_dropout"] == "cpu",
                )
                self.model_filename_list += [(self.src_dst_opt, "src_dst_opt.npy")]

                # 如果使用真实脸部强度，初始化代码鉴别器优化器
                if self.options["true_face_power"] != 0:
                    self.D_code_opt = OptimizerClass(
                        lr=lr,
                        lr_dropout=lr_dropout,
                        lr_cos=lr_cos,
                        clipnorm=clipnorm,
                        name="D_code_opt",
                    )
                    self.D_code_opt.initialize_variables(
                        self.code_discriminator.get_weights(),
                        vars_on_cpu=optimizer_vars_on_cpu,
                        lr_dropout_on_cpu=self.options["lr_dropout"] == "cpu",
                    )
                    self.model_filename_list += [(self.D_code_opt, "D_code_opt.npy")]

                # 如果使用GAN，初始化GAN鉴别器优化器
                if gan_power != 0:
                    self.D_src_dst_opt = OptimizerClass(
                        lr=lr,
                        lr_dropout=lr_dropout,
                        lr_cos=lr_cos,
                        clipnorm=clipnorm,
                        name="GAN_opt",
                    )
                    self.D_src_dst_opt.initialize_variables(
                        self.D_src.get_weights(),
                        vars_on_cpu=optimizer_vars_on_cpu,
                        lr_dropout_on_cpu=self.options["lr_dropout"] == "cpu",
                    )  # +self.D_src_x2.get_weights()
                    self.model_filename_list += [(self.D_src_dst_opt, "GAN_opt.npy")]

        if self.is_training:
            # 调整多GPU环境下的批处理大小
            gpu_count = max(1, len(devices))  # 获取GPU数量，至少为1
            bs_per_gpu = max(
                1, self.get_batch_size() // gpu_count
            )  # 每个GPU的批处理大小，至少为1
            self.set_batch_size(gpu_count * bs_per_gpu)  # 设置总的批处理大小

            # 计算每个GPU的损失
            gpu_pred_src_src_list = []  # GPU预测源到源的列表
            gpu_pred_dst_dst_list = []  # GPU预测目标到目标的列表
            gpu_pred_src_dst_list = []  # GPU预测源到目标的列表
            gpu_pred_src_srcm_list = []  # GPU预测源到源掩码的列表
            gpu_pred_dst_dstm_list = []  # GPU预测目标到目标掩码的列表
            gpu_pred_src_dstm_list = []  # GPU预测源到目标掩码的列表

            gpu_src_losses = []  # GPU源损失列表
            gpu_dst_losses = []  # GPU目标损失列表
            gpu_G_loss_gvs = []  # GPU生成器损失梯度列表
            gpu_D_code_loss_gvs = []  # GPU判别器编码损失梯度列表
            gpu_D_src_dst_loss_gvs = []  # GPU判别器源到目标损失梯度列表

            for gpu_id in range(gpu_count):
                with tf.device(
                    f"/{devices[gpu_id].tf_dev_type}:{gpu_id}"
                    if len(devices) != 0
                    else f"/CPU:0"
                ):
                    with tf.device(f"/CPU:0"):
                        # 在CPU上进行切片操作，以避免所有批处理数据首先被传输到GPU
                        batch_slice = slice(
                            gpu_id * bs_per_gpu, (gpu_id + 1) * bs_per_gpu
                        )
                        gpu_warped_src = self.warped_src[
                            batch_slice, :, :, :
                        ]  # 切片后的变形源图像
                        gpu_warped_dst = self.warped_dst[
                            batch_slice, :, :, :
                        ]  # 切片后的变形目标图像
                        gpu_target_src = self.target_src[
                            batch_slice, :, :, :
                        ]  # 切片后的目标源图像
                        gpu_target_dst = self.target_dst[
                            batch_slice, :, :, :
                        ]  # 切片后的目标目标图像
                        gpu_target_srcm_all = self.target_srcm[
                            batch_slice, :, :, :
                        ]  # 切片后的目标源掩码
                        gpu_target_srcm_em = self.target_srcm_em[
                            batch_slice, :, :, :
                        ]  # 切片后的目标源紧急掩码
                        gpu_target_dstm_all = self.target_dstm[
                            batch_slice, :, :, :
                        ]  # 切片后的目标目标掩码
                        gpu_target_dstm_em = self.target_dstm_em[
                            batch_slice, :, :, :
                        ]  # 切片后的目标目标紧急掩码

                    gpu_target_srcm_anti = 1 - gpu_target_srcm_all  # 源反向掩码
                    gpu_target_dstm_anti = 1 - gpu_target_dstm_all  # 目标反向掩码

                    if blur_out_mask:
                        sigma = resolution / 128

                        x = nn.gaussian_blur(
                            gpu_target_src * gpu_target_srcm_anti, sigma
                        )
                        y = 1 - nn.gaussian_blur(gpu_target_srcm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_src = (
                            gpu_target_src * gpu_target_srcm_all
                            + (x / y) * gpu_target_srcm_anti
                        )

                        x = nn.gaussian_blur(
                            gpu_target_dst * gpu_target_dstm_anti, sigma
                        )
                        y = 1 - nn.gaussian_blur(gpu_target_dstm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_dst = (
                            gpu_target_dst * gpu_target_dstm_all
                            + (x / y) * gpu_target_dstm_anti
                        )

                    # 处理模型张量
                    if "df" in archi_type:
                        # 使用'df'架构类型
                        gpu_src_code = self.inter(self.encoder(gpu_warped_src))
                        gpu_dst_code = self.inter(self.encoder(gpu_warped_dst))
                        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(
                            gpu_src_code
                        )
                        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(
                            gpu_dst_code
                        )
                        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(
                            gpu_dst_code
                        )
                        gpu_pred_src_dst_no_code_grad, _ = self.decoder_src(
                            tf.stop_gradient(gpu_dst_code)
                        )

                    elif "liae" in archi_type:
                        # 使用'liae'架构类型
                        gpu_src_code = self.encoder(gpu_warped_src)
                        gpu_src_inter_AB_code = self.inter_AB(gpu_src_code)
                        gpu_src_code = tf.concat(
                            [gpu_src_inter_AB_code, gpu_src_inter_AB_code],
                            nn.conv2d_ch_axis,
                        )
                        gpu_dst_code = self.encoder(gpu_warped_dst)
                        gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
                        gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
                        gpu_dst_code = tf.concat(
                            [gpu_dst_inter_B_code, gpu_dst_inter_AB_code],
                            nn.conv2d_ch_axis,
                        )
                        gpu_src_dst_code = tf.concat(
                            [gpu_dst_inter_AB_code, gpu_dst_inter_AB_code],
                            nn.conv2d_ch_axis,
                        )

                        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
                        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
                        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(
                            gpu_src_dst_code
                        )
                        gpu_pred_src_dst_no_code_grad, _ = self.decoder(
                            tf.stop_gradient(gpu_src_dst_code)
                        )

                    gpu_pred_src_src_list.append(
                        gpu_pred_src_src
                    )  # 将GPU预测的源到源结果添加到列表
                    gpu_pred_dst_dst_list.append(
                        gpu_pred_dst_dst
                    )  # 将GPU预测的目标到目标结果添加到列表
                    gpu_pred_src_dst_list.append(
                        gpu_pred_src_dst
                    )  # 将GPU预测的源到目标结果添加到列表

                    gpu_pred_src_srcm_list.append(
                        gpu_pred_src_srcm
                    )  # 将GPU预测的源到源掩码添加到列表
                    gpu_pred_dst_dstm_list.append(
                        gpu_pred_dst_dstm
                    )  # 将GPU预测的目标到目标掩码添加到列表
                    gpu_pred_src_dstm_list.append(
                        gpu_pred_src_dstm
                    )  # 将GPU预测的源到目标掩码添加到列表

                    # 从一个组合掩码中解包各个掩码
                    gpu_target_srcm = tf.clip_by_value(
                        gpu_target_srcm_all, 0, 1
                    )  # GPU源掩码
                    gpu_target_dstm = tf.clip_by_value(
                        gpu_target_dstm_all, 0, 1
                    )  # GPU目标掩码
                    gpu_target_srcm_eye_mouth = tf.clip_by_value(
                        gpu_target_srcm_em - 1, 0, 1
                    )  # GPU源眼睛嘴巴掩码
                    gpu_target_dstm_eye_mouth = tf.clip_by_value(
                        gpu_target_dstm_em - 1, 0, 1
                    )  # GPU目标眼睛嘴巴掩码
                    gpu_target_srcm_mouth = tf.clip_by_value(
                        gpu_target_srcm_em - 2, 0, 1
                    )  # GPU源嘴巴掩码
                    gpu_target_dstm_mouth = tf.clip_by_value(
                        gpu_target_dstm_em - 2, 0, 1
                    )  # GPU目标嘴巴掩码
                    gpu_target_srcm_eyes = tf.clip_by_value(
                        gpu_target_srcm_eye_mouth - gpu_target_srcm_mouth, 0, 1
                    )  # GPU源眼睛掩码
                    gpu_target_dstm_eyes = tf.clip_by_value(
                        gpu_target_dstm_eye_mouth - gpu_target_dstm_mouth, 0, 1
                    )  # GPU目标眼睛掩码

                    gpu_target_srcm_blur = nn.gaussian_blur(
                        gpu_target_srcm, max(1, resolution // 32)
                    )  # 模糊处理GPU源掩码
                    gpu_target_srcm_blur = (
                        tf.clip_by_value(gpu_target_srcm_blur, 0, 0.5) * 2
                    )
                    gpu_target_srcm_anti_blur = (
                        1.0 - gpu_target_srcm_blur
                    )  # 反向模糊处理GPU源掩码

                    gpu_target_dstm_blur = nn.gaussian_blur(
                        gpu_target_dstm, max(1, resolution // 32)
                    )  # 模糊处理GPU目标掩码
                    gpu_target_dstm_blur = (
                        tf.clip_by_value(gpu_target_dstm_blur, 0, 0.5) * 2
                    )

                    gpu_style_mask_blur = nn.gaussian_blur(
                        gpu_pred_src_dstm * gpu_pred_dst_dstm, max(1, resolution // 32)
                    )  # 模糊处理风格掩码
                    gpu_style_mask_blur = tf.stop_gradient(
                        tf.clip_by_value(gpu_target_srcm_blur, 0, 1.0)
                    )
                    gpu_style_mask_anti_blur = (
                        1.0 - gpu_style_mask_blur
                    )  # 反向模糊处理风格掩码

                    gpu_target_dst_masked = (
                        gpu_target_dst * gpu_target_dstm_blur
                    )  # 应用模糊处理的GPU目标掩码

                    gpu_target_src_anti_masked = (
                        gpu_target_src * gpu_target_srcm_anti_blur
                    )  # 应用反向模糊处理的GPU源图像
                    gpu_pred_src_src_anti_masked = (
                        gpu_pred_src_src * gpu_target_srcm_anti_blur
                    )  # 应用反向模糊处理的GPU源预测图像

                    gpu_target_src_masked_opt = (
                        gpu_target_src * gpu_target_srcm_blur
                        if masked_training
                        else gpu_target_src
                    )  # 根据是否掩码训练选择GPU源图像
                    gpu_target_dst_masked_opt = (
                        gpu_target_dst_masked if masked_training else gpu_target_dst
                    )  # 根据是否掩码训练选择GPU目标图像
                    gpu_pred_src_src_masked_opt = (
                        gpu_pred_src_src * gpu_target_srcm_blur
                        if masked_training
                        else gpu_pred_src_src
                    )  # 根据是否掩码训练选择GPU源预测图像
                    gpu_pred_dst_dst_masked_opt = (
                        gpu_pred_dst_dst * gpu_target_dstm_blur
                        if masked_training
                        else gpu_pred_dst_dst
                    )  # 根据是否掩码训练选择GPU目标预测图像

                    if self.options["loss_function"] == "MS-SSIM":
                        # 使用MS-SSIM损失函数
                        gpu_src_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                            gpu_target_src_masked_opt,
                            gpu_pred_src_src_masked_opt,
                            max_val=1.0,
                        )
                        gpu_src_loss += tf.reduce_mean(
                            10
                            * tf.square(
                                gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt
                            ),
                            axis=[1, 2, 3],
                        )
                    elif self.options["loss_function"] == "MS-SSIM+L1":
                        # 使用MS-SSIM+L1损失函数
                        gpu_src_loss = 10 * nn.MsSsim(
                            bs_per_gpu, input_ch, resolution, use_l1=True
                        )(
                            gpu_target_src_masked_opt,
                            gpu_pred_src_src_masked_opt,
                            max_val=1.0,
                        )
                    else:
                        # 使用其他损失函数
                        if resolution < 256:
                            gpu_src_loss = tf.reduce_mean(
                                10
                                * nn.dssim(
                                    gpu_target_src_masked_opt,
                                    gpu_pred_src_src_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 11.6),
                                ),
                                axis=[1],
                            )
                        else:
                            gpu_src_loss = tf.reduce_mean(
                                5
                                * nn.dssim(
                                    gpu_target_src_masked_opt,
                                    gpu_pred_src_src_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 11.6),
                                ),
                                axis=[1],
                            )
                            gpu_src_loss += tf.reduce_mean(
                                5
                                * nn.dssim(
                                    gpu_target_src_masked_opt,
                                    gpu_pred_src_src_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 23.2),
                                ),
                                axis=[1],
                            )
                        gpu_src_loss += tf.reduce_mean(
                            10
                            * tf.square(
                                gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt
                            ),
                            axis=[1, 2, 3],
                        )

                    if eyes_prio or mouth_prio:
                        # 如果设置了眼睛或嘴巴优先级
                        if eyes_prio and mouth_prio:
                            gpu_target_part_mask = gpu_target_srcm_eye_mouth
                        elif eyes_prio:
                            gpu_target_part_mask = gpu_target_srcm_eyes
                        elif mouth_prio:
                            gpu_target_part_mask = gpu_target_srcm_mouth

                        gpu_src_loss += tf.reduce_mean(
                            300
                            * tf.abs(
                                gpu_target_src * gpu_target_part_mask
                                - gpu_pred_src_src * gpu_target_part_mask
                            ),
                            axis=[1, 2, 3],
                        )

                    gpu_src_loss += tf.reduce_mean(
                        10 * tf.square(gpu_target_srcm - gpu_pred_src_srcm),
                        axis=[1, 2, 3],
                    )  # 计算GPU源掩码和GPU源预测掩码之间的损失

                    if self.options["background_power"] > 0:
                        bg_factor = self.options["background_power"]

                        if self.options["loss_function"] == "MS-SSIM":
                            gpu_src_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                                    gpu_target_src, gpu_pred_src_src, max_val=1.0
                                )
                            )
                            gpu_src_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_src - gpu_pred_src_src),
                                axis=[1, 2, 3],
                            )
                        elif self.options["loss_function"] == "MS-SSIM+L1":
                            gpu_src_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(
                                    bs_per_gpu, input_ch, resolution, use_l1=True
                                )(gpu_target_src, gpu_pred_src_src, max_val=1.0)
                            )
                        else:
                            if resolution < 256:
                                gpu_src_loss += bg_factor * tf.reduce_mean(
                                    10
                                    * nn.dssim(
                                        gpu_target_src,
                                        gpu_pred_src_src,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                            else:
                                gpu_src_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_src,
                                        gpu_pred_src_src,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                                gpu_src_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_src,
                                        gpu_pred_src_src,
                                        max_val=1.0,
                                        filter_size=int(resolution / 23.2),
                                    ),
                                    axis=[1],
                                )
                            gpu_src_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_src - gpu_pred_src_src),
                                axis=[1, 2, 3],
                            )

                    face_style_power = self.options["face_style_power"] / 100.0
                    if face_style_power != 0 and not self.pretrain:
                        gpu_src_loss += nn.style_loss(
                            gpu_pred_src_dst_no_code_grad
                            * tf.stop_gradient(gpu_pred_src_dstm),
                            tf.stop_gradient(gpu_pred_dst_dst * gpu_pred_dst_dstm),
                            gaussian_blur_radius=resolution // 8,
                            loss_weight=10000 * face_style_power,
                        )

                    bg_style_power = self.options["bg_style_power"] / 100.0
                    if bg_style_power != 0 and not self.pretrain:
                        gpu_target_dst_style_anti_masked = (
                            gpu_target_dst * gpu_style_mask_anti_blur
                        )
                        gpu_psd_style_anti_masked = (
                            gpu_pred_src_dst * gpu_style_mask_anti_blur
                        )

                        gpu_src_loss += tf.reduce_mean(
                            (10 * bg_style_power)
                            * nn.dssim(
                                gpu_psd_style_anti_masked,
                                gpu_target_dst_style_anti_masked,
                                max_val=1.0,
                                filter_size=int(resolution / 11.6),
                            ),
                            axis=[1],
                        )
                        gpu_src_loss += tf.reduce_mean(
                            (10 * bg_style_power)
                            * tf.square(
                                gpu_psd_style_anti_masked
                                - gpu_target_dst_style_anti_masked
                            ),
                            axis=[1, 2, 3],
                        )

                    if self.options["loss_function"] == "MS-SSIM":
                        gpu_dst_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                            gpu_target_dst_masked_opt,
                            gpu_pred_dst_dst_masked_opt,
                            max_val=1.0,
                        )
                        gpu_dst_loss += tf.reduce_mean(
                            10
                            * tf.square(
                                gpu_target_dst_masked_opt - gpu_pred_dst_dst_masked_opt
                            ),
                            axis=[1, 2, 3],
                        )
                    elif self.options["loss_function"] == "MS-SSIM+L1":
                        gpu_dst_loss = 10 * nn.MsSsim(
                            bs_per_gpu, input_ch, resolution, use_l1=True
                        )(
                            gpu_target_dst_masked_opt,
                            gpu_pred_dst_dst_masked_opt,
                            max_val=1.0,
                        )
                    else:
                        if resolution < 256:
                            gpu_dst_loss = tf.reduce_mean(
                                10
                                * nn.dssim(
                                    gpu_target_dst_masked_opt,
                                    gpu_pred_dst_dst_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 11.6),
                                ),
                                axis=[1],
                            )
                        else:
                            gpu_dst_loss = tf.reduce_mean(
                                5
                                * nn.dssim(
                                    gpu_target_dst_masked_opt,
                                    gpu_pred_dst_dst_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 11.6),
                                ),
                                axis=[1],
                            )
                            gpu_dst_loss += tf.reduce_mean(
                                5
                                * nn.dssim(
                                    gpu_target_dst_masked_opt,
                                    gpu_pred_dst_dst_masked_opt,
                                    max_val=1.0,
                                    filter_size=int(resolution / 23.2),
                                ),
                                axis=[1],
                            )
                        gpu_dst_loss += tf.reduce_mean(
                            10
                            * tf.square(
                                gpu_target_dst_masked_opt - gpu_pred_dst_dst_masked_opt
                            ),
                            axis=[1, 2, 3],
                        )

                    if eyes_prio or mouth_prio:
                        if eyes_prio and mouth_prio:
                            gpu_target_part_mask = gpu_target_dstm_eye_mouth
                        elif eyes_prio:
                            gpu_target_part_mask = gpu_target_dstm_eyes
                        elif mouth_prio:
                            gpu_target_part_mask = gpu_target_dstm_mouth

                        gpu_dst_loss += tf.reduce_mean(
                            300
                            * tf.abs(
                                gpu_target_dst * gpu_target_part_mask
                                - gpu_pred_dst_dst * gpu_target_part_mask
                            ),
                            axis=[1, 2, 3],
                        )

                    if self.options["background_power"] > 0:
                        bg_factor = self.options["background_power"]

                        if self.options["loss_function"] == "MS-SSIM":
                            gpu_dst_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(bs_per_gpu, input_ch, resolution)(
                                    gpu_target_dst, gpu_pred_dst_dst, max_val=1.0
                                )
                            )
                            gpu_dst_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_dst - gpu_pred_dst_dst),
                                axis=[1, 2, 3],
                            )
                        elif self.options["loss_function"] == "MS-SSIM+L1":
                            gpu_dst_loss += (
                                bg_factor
                                * 10
                                * nn.MsSsim(
                                    bs_per_gpu, input_ch, resolution, use_l1=True
                                )(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0)
                            )
                        else:
                            if resolution < 256:
                                gpu_dst_loss += bg_factor * tf.reduce_mean(
                                    10
                                    * nn.dssim(
                                        gpu_target_dst,
                                        gpu_pred_dst_dst,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                            else:
                                gpu_dst_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_dst,
                                        gpu_pred_dst_dst,
                                        max_val=1.0,
                                        filter_size=int(resolution / 11.6),
                                    ),
                                    axis=[1],
                                )
                                gpu_dst_loss += bg_factor * tf.reduce_mean(
                                    5
                                    * nn.dssim(
                                        gpu_target_dst,
                                        gpu_pred_dst_dst,
                                        max_val=1.0,
                                        filter_size=int(resolution / 23.2),
                                    ),
                                    axis=[1],
                                )
                            gpu_dst_loss += bg_factor * tf.reduce_mean(
                                10 * tf.square(gpu_target_dst - gpu_pred_dst_dst),
                                axis=[1, 2, 3],
                            )

                    gpu_dst_loss += tf.reduce_mean(
                        10 * tf.square(gpu_target_dstm - gpu_pred_dst_dstm),
                        axis=[1, 2, 3],
                    )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss

                    def DLoss(labels, logits):
                        return tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=labels, logits=logits
                            ),
                            axis=[1, 2, 3],
                        )

                    if self.options["true_face_power"] != 0:
                        gpu_src_code_d = self.code_discriminator(gpu_src_code)
                        gpu_src_code_d_ones = tf.ones_like(gpu_src_code_d)
                        gpu_src_code_d_zeros = tf.zeros_like(gpu_src_code_d)
                        gpu_dst_code_d = self.code_discriminator(gpu_dst_code)
                        gpu_dst_code_d_ones = tf.ones_like(gpu_dst_code_d)

                        gpu_G_loss += self.options["true_face_power"] * DLoss(
                            gpu_src_code_d_ones, gpu_src_code_d
                        )

                        gpu_D_code_loss = (
                            DLoss(gpu_dst_code_d_ones, gpu_dst_code_d)
                            + DLoss(gpu_src_code_d_zeros, gpu_src_code_d)
                        ) * 0.5

                        gpu_D_code_loss_gvs += [
                            nn.gradients(
                                gpu_D_code_loss, self.code_discriminator.get_weights()
                            )
                        ]

                    if gan_power != 0:
                        gpu_pred_src_src_d, gpu_pred_src_src_d2 = self.D_src(
                            gpu_pred_src_src_masked_opt
                        )

                        def get_smooth_noisy_labels(
                            label, tensor, smoothing=0.1, noise=0.05
                        ):
                            num_labels = self.batch_size
                            for d in tensor.get_shape().as_list()[1:]:
                                num_labels *= d

                            probs = (
                                tf.math.log([[noise, 1 - noise]])
                                if label == 1
                                else tf.math.log([[1 - noise, noise]])
                            )
                            x = tf.random.categorical(probs, num_labels)
                            x = tf.cast(x, tf.float32)
                            x = tf.math.scalar_mul(1 - smoothing, x)
                            # x = x + (smoothing/num_labels)
                            x = tf.reshape(
                                x,
                                (self.batch_size,)
                                + tuple(tensor.get_shape().as_list()[1:]),
                            )
                            return x

                        smoothing = self.options["gan_smoothing"]
                        noise = self.options["gan_noise"]

                        gpu_pred_src_src_d_ones = tf.ones_like(gpu_pred_src_src_d)
                        gpu_pred_src_src_d2_ones = tf.ones_like(gpu_pred_src_src_d2)

                        gpu_pred_src_src_d_smooth_zeros = get_smooth_noisy_labels(
                            0, gpu_pred_src_src_d, smoothing=smoothing, noise=noise
                        )
                        gpu_pred_src_src_d2_smooth_zeros = get_smooth_noisy_labels(
                            0, gpu_pred_src_src_d2, smoothing=smoothing, noise=noise
                        )

                        gpu_target_src_d, gpu_target_src_d2 = self.D_src(
                            gpu_target_src_masked_opt
                        )

                        gpu_target_src_d_smooth_ones = get_smooth_noisy_labels(
                            1, gpu_target_src_d, smoothing=smoothing, noise=noise
                        )
                        gpu_target_src_d2_smooth_ones = get_smooth_noisy_labels(
                            1, gpu_target_src_d2, smoothing=smoothing, noise=noise
                        )

                        gpu_D_src_dst_loss = (
                            DLoss(gpu_target_src_d_smooth_ones, gpu_target_src_d)
                            + DLoss(gpu_pred_src_src_d_smooth_zeros, gpu_pred_src_src_d)
                            + DLoss(gpu_target_src_d2_smooth_ones, gpu_target_src_d2)
                            + DLoss(
                                gpu_pred_src_src_d2_smooth_zeros, gpu_pred_src_src_d2
                            )
                        )

                        gpu_D_src_dst_loss_gvs += [
                            nn.gradients(gpu_D_src_dst_loss, self.D_src.get_weights())
                        ]  # +self.D_src_x2.get_weights()

                        gpu_G_loss += gan_power * (
                            DLoss(gpu_pred_src_src_d_ones, gpu_pred_src_src_d)
                            + DLoss(gpu_pred_src_src_d2_ones, gpu_pred_src_src_d2)
                        )

                        if masked_training:
                            # Minimal src-src-bg rec with total_variation_mse to suppress random bright dots from gan
                            gpu_G_loss += 0.000001 * nn.total_variation_mse(
                                gpu_pred_src_src
                            )
                            gpu_G_loss += 0.02 * tf.reduce_mean(
                                tf.square(
                                    gpu_pred_src_src_anti_masked
                                    - gpu_target_src_anti_masked
                                ),
                                axis=[1, 2, 3],
                            )

                    gpu_G_loss_gvs += [
                        nn.gradients(gpu_G_loss, self.src_dst_trainable_weights)
                    ]

            # Average losses and gradients, and create optimizer update ops
            with tf.device(f"/CPU:0"):
                pred_src_src = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

            with tf.device(models_opt_device):
                src_loss = tf.concat(gpu_src_losses, 0)
                dst_loss = tf.concat(gpu_dst_losses, 0)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op(
                    nn.average_gv_list(gpu_G_loss_gvs)
                )

                if self.options["true_face_power"] != 0:
                    D_loss_gv_op = self.D_code_opt.get_update_op(
                        nn.average_gv_list(gpu_D_code_loss_gvs)
                    )

                if gan_power != 0:
                    src_D_src_dst_loss_gv_op = self.D_src_dst_opt.get_update_op(
                        nn.average_gv_list(gpu_D_src_dst_loss_gvs)
                    )

            # Initializing training and view functions
            def src_dst_train(
                warped_src,
                target_src,
                target_srcm,
                target_srcm_em,
                warped_dst,
                target_dst,
                target_dstm,
                target_dstm_em,
            ):
                s, d = nn.tf_sess.run(
                    [src_loss, dst_loss, src_dst_loss_gv_op],
                    feed_dict={
                        self.warped_src: warped_src,
                        self.target_src: target_src,
                        self.target_srcm: target_srcm,
                        self.target_srcm_em: target_srcm_em,
                        self.warped_dst: warped_dst,
                        self.target_dst: target_dst,
                        self.target_dstm: target_dstm,
                        self.target_dstm_em: target_dstm_em,
                    },
                )[:2]
                return s, d

            self.src_dst_train = src_dst_train

            def get_src_dst_information(
                warped_src,
                target_src,
                target_srcm,
                target_srcm_em,
                warped_dst,
                target_dst,
                target_dstm,
                target_dstm_em,
            ):
                out_data = nn.tf_sess.run(
                    [
                        src_loss,
                        dst_loss,
                        pred_src_src,
                        pred_src_srcm,
                        pred_dst_dst,
                        pred_dst_dstm,
                        pred_src_dst,
                        pred_src_dstm,
                    ],
                    feed_dict={
                        self.warped_src: warped_src,
                        self.target_src: target_src,
                        self.target_srcm: target_srcm,
                        self.target_srcm_em: target_srcm_em,
                        self.warped_dst: warped_dst,
                        self.target_dst: target_dst,
                        self.target_dstm: target_dstm,
                        self.target_dstm_em: target_dstm_em,
                    },
                )

                return out_data

            self.get_src_dst_information = get_src_dst_information

            if self.options["true_face_power"] != 0:

                def D_train(warped_src, warped_dst):
                    nn.tf_sess.run(
                        [D_loss_gv_op],
                        feed_dict={
                            self.warped_src: warped_src,
                            self.warped_dst: warped_dst,
                        },
                    )

                self.D_train = D_train

            if gan_power != 0:

                def D_src_dst_train(
                    warped_src,
                    target_src,
                    target_srcm,
                    target_srcm_em,
                    warped_dst,
                    target_dst,
                    target_dstm,
                    target_dstm_em,
                ):
                    nn.tf_sess.run(
                        [src_D_src_dst_loss_gv_op],
                        feed_dict={
                            self.warped_src: warped_src,
                            self.target_src: target_src,
                            self.target_srcm: target_srcm,
                            self.target_srcm_em: target_srcm_em,
                            self.warped_dst: warped_dst,
                            self.target_dst: target_dst,
                            self.target_dstm: target_dstm,
                            self.target_dstm_em: target_dstm_em,
                        },
                    )

                self.D_src_dst_train = D_src_dst_train

            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run(
                    [
                        pred_src_src,
                        pred_src_srcm,
                        pred_dst_dst,
                        pred_dst_dstm,
                        pred_src_dst,
                        pred_src_dstm,
                    ],
                    feed_dict={
                        self.warped_src: warped_src,
                        self.warped_dst: warped_dst,
                    },
                )

            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device(
                nn.tf_default_device_name if len(devices) != 0 else f"/CPU:0"
            ):
                if "df" in archi_type:
                    gpu_dst_code = self.inter(self.encoder(self.warped_dst))
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                    _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

                elif "liae" in archi_type:
                    gpu_dst_code = self.encoder(self.warped_dst)
                    gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
                    gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
                    gpu_dst_code = tf.concat(
                        [gpu_dst_inter_B_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis
                    )
                    gpu_src_dst_code = tf.concat(
                        [gpu_dst_inter_AB_code, gpu_dst_inter_AB_code],
                        nn.conv2d_ch_axis,
                    )

                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                    _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)

            def AE_merge(warped_dst):
                return nn.tf_sess.run(
                    [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm],
                    feed_dict={self.warped_dst: warped_dst},
                )

            self.AE_merge = AE_merge

        # 遍历模型和对应的文件名
        for model, filename in io.progress_bar_generator(
            self.model_filename_list, "初始化模型"
        ):
            # 检查预训练是否刚刚被禁用
            if self.pretrain_just_disabled:
                do_init = False
                # 如果架构类型包含"df"
                if "df" in archi_type:
                    # 对于特定的模型，需要进行初始化
                    if model == self.inter:
                        print("预训练转正...")
                        ask_for_clean = input(
                            "是否重置inter？（重置后效果更好，但训练更慢）请输入 'y' 或 'n': "
                        )
                        if ask_for_clean.lower() == "y":
                            do_init = True
                            print("inter已重置！")
                        else:
                            do_init = False
                            print("保留inter继续训练！建议开启随机扭曲！")

                # 如果架构类型是"liae"
                elif "liae" in archi_type:
                    # 对于特定的模型，需要进行初始化
                    if model == self.inter_AB:
                        ask_for_clean = input(
                            "是否重置inter_AB？（重置后效果更好，但训练更慢）请输入 'y' 或 'n': "
                        )
                        if ask_for_clean.lower() == "y":
                            do_init = True
                            print("inter_AB已重置！")
                        else:
                            do_init = False
                            print("保留inter_AB继续训练！建议开启随机扭曲！")
            else:
                # 检查是否是第一次运行
                do_init = self.is_first_run()
                # 如果是训练模式，并且GAN的能力不为0，对特定的模型进行初始化
                if self.is_training and gan_power != 0 and model == self.D_src:
                    if self.gan_model_changed:
                        do_init = True

            # 如果不需要初始化，则尝试从文件加载模型权重
            if not do_init:
                do_init = not model.load_weights(
                    self.get_strpath_storage_for_file(filename)
                )

            # 如果需要初始化，初始化模型权重
            if do_init:
                model.init_weights()

        ###############
        # 初始化样本生成器
        if self.is_training:
            # 如果是在训练模式下
            training_data_src_path = (
                self.training_data_src_path  # 使用指定的训练数据源路径
                if not self.pretrain  # 如果不是在预训练模式
                else self.get_pretraining_data_path()  # 否则使用预训练数据路径
            )
            training_data_dst_path = (
                self.training_data_dst_path  # 使用指定的目标训练数据路径
                if not self.pretrain  # 如果不是在预训练模式
                else self.get_pretraining_data_path()  # 否则使用预训练数据路径
            )
            # 如果指定了ct_mode并且不是预训练模式，则使用目标训练数据路径
            random_ct_samples_path = (
                training_data_dst_path
                if ct_mode is not None and not self.pretrain
                else None  # 否则不使用任何路径
            )

            # 获取CPU核心数，但不超过设定的上限
            cpu_count = min(multiprocessing.cpu_count(), self.options["cpu_cap"])
            src_generators_count = cpu_count // 2  # 源数据生成器的数量为CPU核心数的一半
            dst_generators_count = (
                cpu_count // 2
            )  # 目标数据生成器的数量也是CPU核心数的一半
            if ct_mode is not None:
                src_generators_count = int(
                    src_generators_count * 1.5
                )  # 如果指定了ct_mode，则增加源数据生成器的数量

            dst_aug = None  # 初始化目标数据增强为None
            allowed_dst_augs = ["fs-aug", "cc-aug"]  # 定义允许的目标数据增强类型
            if ct_mode in allowed_dst_augs:
                dst_aug = ct_mode  # 如果ct_mode是允许的类型，则使用该类型的数据增强

            channel_type = (
                SampleProcessor.ChannelType.LAB_RAND_TRANSFORM  # 如果开启了随机颜色选项
                if self.options["random_color"]
                else SampleProcessor.ChannelType.BGR  # 否则使用BGR通道类型
            )

            # Check for pak names
            # give priority to pak names in configuration file
            if self.read_from_conf and self.config_file_exists:
                conf_src_pak_name = self.options.get("src_pak_name", None)
                conf_dst_pak_name = self.options.get("dst_pak_name", None)
                if conf_src_pak_name is not None:
                    self.src_pak_name = conf_src_pak_name
                if conf_dst_pak_name is not None:
                    self.dst_pak_name = conf_dst_pak_name

            ignore_same_path = False
            if (
                self.src_pak_name != self.dst_pak_name
                and training_data_src_path == training_data_dst_path
                and not self.pretrain
            ):
                ignore_same_path = True
            elif self.pretrain:
                self.src_pak_name = self.dst_pak_name = "faceset"

            # print("test super warp",self.rotation_range,self.scale_range)
            self.set_training_data_generators(
                [
                    SampleGeneratorFace(
                        training_data_src_path,
                        pak_name=self.src_pak_name,
                        ignore_same_path=ignore_same_path,
                        random_ct_samples_path=random_ct_samples_path,
                        debug=self.is_debug(),
                        batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(
                            rotation_range=self.rotation_range,
                            scale_range=self.scale_range,
                            random_flip=random_src_flip,
                        ),
                        output_sample_types=[
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": random_warp,
                                "random_downsample": self.options["random_downsample"],
                                "random_noise": self.options["random_noise"],
                                "random_blur": self.options["random_blur"],
                                "random_jpeg": self.options["random_jpeg"],
                                "random_hsv_shift_amount": random_hsv_power,
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": ct_mode,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": False,
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": ct_mode,
                                "random_hsv_shift_amount": random_hsv_power,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.EYES_MOUTH,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                        ],
                        uniform_yaw_distribution=self.options["uniform_yaw"]
                        or self.pretrain,
                        generators_count=src_generators_count,
                    ),
                    SampleGeneratorFace(
                        training_data_dst_path,
                        pak_name=self.dst_pak_name,
                        ignore_same_path=ignore_same_path,
                        debug=self.is_debug(),
                        batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(
                            rotation_range=self.rotation_range,
                            scale_range=self.scale_range,
                            random_flip=random_src_flip,
                        ),
                        output_sample_types=[
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": random_warp,
                                "random_downsample": self.options["random_downsample"],
                                "random_noise": self.options["random_noise"],
                                "random_blur": self.options["random_blur"],
                                "random_jpeg": self.options["random_jpeg"],
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": dst_aug,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                                "warp": False,
                                "transform": True,
                                "channel_type": channel_type,
                                "ct_mode": dst_aug,
                                "random_hsv_shift_amount": random_hsv_power,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                            {
                                "sample_type": SampleProcessor.SampleType.FACE_MASK,
                                "warp": False,
                                "transform": True,
                                "channel_type": SampleProcessor.ChannelType.G,
                                "face_mask_type": SampleProcessor.FaceMaskType.EYES_MOUTH,
                                "face_type": self.face_type,
                                "data_format": nn.data_format,
                                "resolution": resolution,
                            },
                        ],
                        uniform_yaw_distribution=self.options["uniform_yaw"]
                        or self.pretrain,
                        generators_count=dst_generators_count,
                    ),
                ]
            )

            if self.options["retraining_samples"]:
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

            if self.pretrain_just_disabled:
                self.update_sample_for_preview(force_new=True)

    def export_dfm(self):
        output_path = self.get_strpath_storage_for_file("model.dfm")

        io.log_info(f"导出 .dfm 到 {output_path}")

        tf = nn.tf
        nn.set_data_format("NCHW")

        with tf.device(nn.tf_default_device_name):
            warped_dst = tf.placeholder(
                nn.floatx, (None, self.resolution, self.resolution, 3), name="in_face"
            )
            warped_dst = tf.transpose(warped_dst, (0, 3, 1, 2))

            if "df" in self.archi_type:
                gpu_dst_code = self.inter(self.encoder(warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            elif "liae" in self.archi_type:
                gpu_dst_code = self.encoder(warped_dst)
                gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
                gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
                gpu_dst_code = tf.concat(
                    [gpu_dst_inter_B_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis
                )
                gpu_src_dst_code = tf.concat(
                    [gpu_dst_inter_AB_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis
                )

                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)

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
                name="SAEHD512",
                input_names=["in_face:0"],
                output_names=[
                    "out_face_mask:0",
                    "out_celeb_face:0",
                    "out_celeb_face_mask:0",
                ],
                opset=12,
                output_path=output_path,
            )

    # override
    def get_model_filename_list(self):
        return self.model_filename_list

    # override
    def onSave(self):
        for model, filename in io.progress_bar_generator(
            self.get_model_filename_list(), "保存中...", leave=False
        ):
            model.save_weights(self.get_strpath_storage_for_file(filename))

    # override
    def should_save_preview_history(self):
        return (
            not io.is_colab()
            and self.iter % (10 * (max(1, self.resolution // 64))) == 0
        ) or (io.is_colab() and self.iter % 100 == 0)

    # override
    def onTrainOneIter(self):
        if (
            self.is_first_run()
            and not self.pretrain
            and not self.pretrain_just_disabled
        ):
            io.log_info("您正在从头开始训练模型。强烈建议使用预训练模型以提高效率.\n")

        (
            (warped_src, target_src, target_srcm, target_srcm_em),
            (warped_dst, target_dst, target_dstm, target_dstm_em),
        ) = self.generate_next_samples()

        src_loss, dst_loss = self.src_dst_train(
            warped_src,
            target_src,
            target_srcm,
            target_srcm_em,
            warped_dst,
            target_dst,
            target_dstm,
            target_dstm_em,
        )

        if self.options["retraining_samples"]:
            bs = self.get_batch_size()

            for i in range(bs):
                self.last_src_samples_loss.append(
                    (target_src[i], target_srcm[i], target_srcm_em[i], src_loss[i])
                )
                self.last_dst_samples_loss.append(
                    (target_dst[i], target_dstm[i], target_dstm_em[i], dst_loss[i])
                )

            if len(self.last_src_samples_loss) >= bs * 16:
                src_samples_loss = sorted(
                    self.last_src_samples_loss, key=operator.itemgetter(3), reverse=True
                )
                dst_samples_loss = sorted(
                    self.last_dst_samples_loss, key=operator.itemgetter(3), reverse=True
                )

                target_src = np.stack([x[0] for x in src_samples_loss[:bs]])
                target_srcm = np.stack([x[1] for x in src_samples_loss[:bs]])
                target_srcm_em = np.stack([x[2] for x in src_samples_loss[:bs]])

                target_dst = np.stack([x[0] for x in dst_samples_loss[:bs]])
                target_dstm = np.stack([x[1] for x in dst_samples_loss[:bs]])
                target_dstm_em = np.stack([x[2] for x in dst_samples_loss[:bs]])

                src_loss, dst_loss = self.src_dst_train(
                    target_src,
                    target_src,
                    target_srcm,
                    target_srcm_em,
                    target_dst,
                    target_dst,
                    target_dstm,
                    target_dstm_em,
                )
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

        if self.options["true_face_power"] != 0 and not self.pretrain:
            self.D_train(warped_src, warped_dst)

        if self.gan_power != 0:
            self.D_src_dst_train(
                warped_src,
                target_src,
                target_srcm,
                target_srcm_em,
                warped_dst,
                target_dst,
                target_dstm,
                target_dstm_em,
            )

        return (
            ("src_loss", np.mean(src_loss)),
            ("dst_loss", np.mean(dst_loss)),
        )

    # override
    def onGetPreview(self, samples, for_history=False, filenames=None):
        (
            (warped_src, target_src, target_srcm, target_srcm_em),
            (warped_dst, target_dst, target_dstm, target_dstm_em),
        ) = samples

        S, D, SS, SSM, DD, DDM, SD, SDM = [
            np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
            for x in ([target_src, target_dst] + self.AE_view(target_src, target_dst))
        ]
        SW, DW = [
            np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
            for x in ([warped_src, warped_dst])
        ]
        (
            SSM,
            DDM,
            SDM,
        ) = [np.repeat(x, (3,), -1) for x in [SSM, DDM, SDM]]

        target_srcm, target_dstm = [
            nn.to_data_format(x, "NHWC", self.model_data_format)
            for x in ([target_srcm, target_dstm])
        ]

        n_samples = min(self.get_batch_size(), self.options["preview_samples"])

        if filenames is not None and len(filenames) > 0:
            for i in range(n_samples):
                S[i] = label_face_filename(S[i], filenames[0][i])
                D[i] = label_face_filename(D[i], filenames[1][i])

        if self.resolution <= 256 or self.options["force_full_preview"] == True:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i], D[i], DD[i], SD[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN", np.concatenate(st, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = SW[i], SS[i], DW[i], DD[i], SD[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped", np.concatenate(wt, axis=0)),
            ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i] * SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                SM = S[i] * target_srcm[i]
                DM = D[i] * target_dstm[i]
                if filenames is not None and len(filenames) > 0:
                    SM = label_face_filename(SM, filenames[0][i])
                    DM = label_face_filename(DM, filenames[1][i])
                ar = SM, SS[i] * SSM[i], DM, DD[i] * DDM[i], SD[i] * SD_mask
                st_m.append(np.concatenate(ar, axis=1))

            result += [
                ("SN masked", np.concatenate(st_m, axis=0)),
            ]
        else:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN src-src", np.concatenate(st, axis=0)),
            ]

            st = []
            for i in range(n_samples):
                ar = D[i], DD[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN dst-dst", np.concatenate(st, axis=0)),
            ]

            st = []
            for i in range(n_samples):
                ar = D[i], SD[i]
                st.append(np.concatenate(ar, axis=1))
            result += [
                ("SN pred", np.concatenate(st, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = SW[i], SS[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped src-src", np.concatenate(wt, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = DW[i], DD[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped dst-dst", np.concatenate(wt, axis=0)),
            ]

            wt = []
            for i in range(n_samples):
                ar = DW[i], SD[i]
                wt.append(np.concatenate(ar, axis=1))
            result += [
                ("SN warped pred", np.concatenate(wt, axis=0)),
            ]

            st_m = []
            for i in range(n_samples):
                ar = S[i] * target_srcm[i], SS[i] * SSM[i]
                st_m.append(np.concatenate(ar, axis=1))
            result += [
                ("SN masked src-src", np.concatenate(st_m, axis=0)),
            ]

            st_m = []
            for i in range(n_samples):
                ar = D[i] * target_dstm[i], DD[i] * DDM[i]
                st_m.append(np.concatenate(ar, axis=1))
            result += [
                ("SN masked dst-dst", np.concatenate(st_m, axis=0)),
            ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i] * SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                ar = D[i] * target_dstm[i], SD[i] * SD_mask
                st_m.append(np.concatenate(ar, axis=1))
            result += [
                ("SN masked pred", np.concatenate(st_m, axis=0)),
            ]

        return result

    def predictor_func(self, face=None):
        face = nn.to_data_format(face[None, ...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [
            nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32)
            for x in self.AE_merge(face)
        ]

        return bgr[0], mask_src_dstm[0][..., 0], mask_dst_dstm[0][..., 0]

    # override
    def get_MergerConfig(self):
        import merger

        return (
            self.predictor_func,
            (self.options["resolution"], self.options["resolution"], 3),
            merger.MergerConfigMasked(face_type=self.face_type, default_mode="overlay"),
        )

    # override
    def get_config_schema_path(self):
        config_path = Path(__file__).parent.absolute() / Path("config_schema.json")
        return config_path

    # override
    def get_formatted_configuration_path(self):
        config_path = Path(__file__).parent.absolute() / Path("formatted_config.yaml")
        return config_path

    # function is WIP
    def generate_training_state(self):
        # 导入所需模块
        from tqdm import tqdm

        import datetime
        import json
        from itertools import zip_longest
        import multiprocessing as mp

        # 生成训练状态
        src_gen = self.generator_list[
            0
        ]  # 获取生成器列表中的第一个生成器对象，赋值给变量src_gen
        dst_gen = self.generator_list[
            1
        ]  # 获取生成器列表中的第二个生成器对象，赋值给变量dst_gen
        self.src_sample_state = []  # 初始化变量self.src_sample_state为空列表
        self.dst_sample_state = []  # 初始化变量self.dst_sample_state为空列表

        src_samples = src_gen.samples  # 获取源生成器的样本
        dst_samples = dst_gen.samples  # 获取目标生成器的样本
        src_len = len(src_samples)  # 获取源样本的长度
        dst_len = len(dst_samples)  # 获取目标样本的长度
        length = src_len  # 初始化长度为源样本的长度
        if length < dst_len:  # 如果源样本的长度小于目标样本的长度
            length = dst_len  # 更新长度为目标样本的长度

        # set paths
        # create core folder
        self.state_history_path = self.saved_models_path / (
            f"{self.get_model_name()}_state_history"
        )
        # 状态历史记录路径为保存模型路径下的特定模型名称加下划线加状态历史记录
        if not self.state_history_path.exists():
            # 如果状态历史记录路径不存在
            self.state_history_path.mkdir(exist_ok=True)
            # 在状态历史记录路径下创建新目录
        # create state folder
        idx_str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")  # 获取当前时间戳
        idx_state_history_path = (
            self.state_history_path / idx_str
        )  # 获取状态历史记录路径
        idx_state_history_path.mkdir()  # 创建状态历史记录路径
        # create set folders
        self.src_state_path = (
            idx_state_history_path / "src"
        )  # 指定源状态路径为索引状态历史路径下的"src"文件夹
        self.src_state_path.mkdir()  # 创建源状态文件夹
        self.dst_state_path = (
            idx_state_history_path / "dst"
        )  # 指定目标状态路径为索引状态历史路径下的"dst"文件夹
        self.dst_state_path.mkdir()  # 创建目标状态文件夹

        print("Generating dataset state snapshot 生成数据集状态快照\r")

        # doing batch 2 always since it is coded to always expect dst and src
        # if one is smaller reapeating the last sample as a placeholder

        # 0 means ignore and use dummy data
        # 将源样本和目标样本进行等长打包，并使用0填充
        data_list = list(zip_longest(src_samples, dst_samples, fillvalue=0))

        # 创建一个全零的虚拟输入数组，形状为(self.resolution, self.resolution, 3)，数据类型为np.float32
        self._dummy_input = np.zeros(
            (self.resolution, self.resolution, 3), dtype=np.float32
        )

        # 创建一个全零的虚拟掩码数组，形状为(self.resolution, self.resolution, 1)，数据类型为np.float32
        self._dummy_mask = np.zeros(
            (self.resolution, self.resolution, 1), dtype=np.float32
        )

        # 对于数据列表中的每个样本元组，使用tqdm库进行迭代
        for sample_tuple in tqdm(data_list, desc="数据下载中", total=len(data_list)):
            # 调用处理器函数处理样本元组
            self._processor(sample_tuple)

        # save model state params
        # copy model summary
        # model_summary = self.options.copy()
        model_summary = {}  # 创建一个空字典，用于存储模型摘要信息
        model_summary["iter"] = (
            self.get_iter()
        )  # 获取当前迭代次数，并将其作为"iter"键的值存储到model_summary字典中
        model_summary["name"] = (
            self.get_model_name()
        )  # 获取模型名称，并将其作为"name"键的值存储到model_summary字典中

        # error with some types, need to double check
        with open(idx_state_history_path / "model_summary.json", "w") as outfile:
            # 使用open函数打开文件model_summary.json，以写入模式写入数据到文件中
            # 文件指针会由with语句自动管理
            json.dump(model_summary, outfile)
            # 将model_summary中的数据以json格式写入outfile文件中

        # training state, full loss stuff from .dat file - prolly should be global
        # state_history_json = self.loss_history

        # main config data
        # set name and full path
        config_dict = {
            "datasets": [
                {"name": "src", "path": str(self.training_data_src_path)},
                {"name": "dst", "path": str(self.training_data_dst_path)},
            ]
        }

        # 创建一个包含训练数据源路径和训练数据目标路径的配置字典
        with open(self.state_history_path / "config.json", "w") as outfile:
            json.dump(config_dict, outfile)
        # save image loss data
        src_full_state_dict = {
            "data": self.src_sample_state,  # 定义一个字典src_full_state_dict，其中键"data"的值为self.src_sample_state
            "set": "src",  # 键"set"的值为"src"
            "type": "set-state",  # 键"type"的值为"set-state"
        }

        with open(
            idx_state_history_path / "src_state.json", "w"
        ) as outfile:  # 打开文件"src_state.json"，以写入方式打开，并将文件对象赋值给outfile
            json.dump(
                src_full_state_dict, outfile
            )  # 将src_full_state_dict以json格式写入outfile

        dst_full_state_dict = {
            "data": self.dst_sample_state,  # 将self.dst_sample_state赋值给键"data"
            "set": "dst",  # 将字符串"dst"赋值给键"set"
            "type": "set-state",  # 将字符串"set-state"赋值给键"type"
        }
        with open(idx_state_history_path / "dst_state.json", "w") as outfile:
            json.dump(dst_full_state_dict, outfile)

        print("完成")

    def _get_formatted_image(self, raw_output):
        # 将原始输出格式转换为指定的数据格式，并进行裁剪，使其值在0到1之间
        formatted = np.clip(
            nn.to_data_format(raw_output, "NHWC", self.model_data_format), 0.0, 1.0
        )
        # 将第一个维度的维度数压缩，得到最终的输出图像
        formatted = np.squeeze(formatted, 0)

        return formatted

    # 导出src dst Loss损失图日志=========================================================================
    def _processor(self, samples_tuple):
        """
        对输入的样本元组进行处理

        Args:
            samples_tuple: 一个包含两个样本的元组，samples_tuple[0]为源样本，samples_tuple[1]为目标样本

        Returns:
            None

        """
        if samples_tuple[0] != 0:
            src_sample_bgr, src_sample_mask, src_sample_mask_em = prepare_sample(
                samples_tuple[0], self.options, self.resolution, self.face_type
            )
        else:
            src_sample_bgr, src_sample_mask, src_sample_mask_em = (
                self._dummy_input,
                self._dummy_mask,
                self._dummy_mask,
            )
        if samples_tuple[1] != 0:
            dst_sample_bgr, dst_sample_mask, dst_sample_mask_em = prepare_sample(
                samples_tuple[1], self.options, self.resolution, self.face_type
            )
        else:
            dst_sample_bgr, dst_sample_mask, dst_sample_mask_em = (
                self._dummy_input,
                self._dummy_mask,
                self._dummy_mask,
            )

        (
            src_loss,  # 源图像损失
            dst_loss,  # 目标图像损失
            pred_src_src,  # 源图像预测的源图像
            pred_src_srcm,  # 源图像预测的源图像混合
            pred_dst_dst,  # 目标图像预测的目标图像
            pred_dst_dstm,  # 目标图像预测的目标图像强度变化
            pred_src_dst,  # 源图像预测的目标图像
            pred_src_dstm,  # 源图像预测的目标图像强度变化
        ) = self.get_src_dst_information(  # 调用get_src_dst_information方法获取源图像和目标图像的相关信息
            data_format_change(
                src_sample_bgr
            ),  # 调用data_format_change方法改变源图像颜色通道的顺序
            data_format_change(
                src_sample_bgr
            ),  # 调用data_format_change方法改变源图像颜色通道的顺序
            data_format_change(
                src_sample_mask
            ),  # 调用data_format_change方法改变源图像掩码的通道顺序
            data_format_change(
                src_sample_mask_em
            ),  # 调用data_format_change方法改变源图像掩码能量的通道顺序
            data_format_change(
                dst_sample_bgr
            ),  # 调用data_format_change方法改变目标图像颜色通道的顺序
            data_format_change(
                dst_sample_bgr
            ),  # 调用data_format_change方法改变目标图像颜色通道的顺序
            data_format_change(
                dst_sample_mask
            ),  # 调用data_format_change方法改变目标图像掩码的通道顺序
            data_format_change(
                dst_sample_mask_em
            ),  # 调用data_format_change方法改变目标图像掩码能量的通道顺序
        )

        if samples_tuple[0] != 0:
            # 获取样本文件名的.stem
            src_file_name = Path(samples_tuple[0].filename).stem

            # 将处理后的图像保存为jpg文件
            cv2_imwrite(
                self.src_state_path / f"{src_file_name}_output.jpg",
                self._get_formatted_image(pred_src_src) * 255,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )  # output

            src_data = {
                # 将src_loss的第一个元素转换为浮点数并赋值给loss键
                "loss": float(src_loss[0]),
                # 将src_file_name加上后缀.jpg并赋值给input键
                "input": f"{src_file_name}.jpg",
                # 将src_file_name加上后缀_output.jpg并赋值给output键
                "output": f"{src_file_name}_output.jpg",
            }
            # 将src_data添加到self.src_sample_state列表中
            self.src_sample_state.append(src_data)

        if samples_tuple[1] != 0:
            # 获取文件名并去扩展名
            dst_file_name = Path(samples_tuple[1].filename).stem

            # 将预测结果保存为图片
            cv2_imwrite(
                self.dst_state_path / f"{dst_file_name}_output.jpg",
                self._get_formatted_image(pred_dst_dst) * 255,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )  # 输出图片
            cv2_imwrite(
                self.dst_state_path / f"{dst_file_name}_swap.jpg",
                self._get_formatted_image(pred_src_dst) * 255,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )  # 交换图片

            # 构建保存结果的数据字典
            dst_data = {
                "loss": float(dst_loss[0]),
                "input": f"{dst_file_name}.jpg",
                "output": f"{dst_file_name}_output.jpg",
                "swap": f"{dst_file_name}_swap.jpg",
            }
            # 将结果数据添加到目标样本状态列表中
            self.dst_sample_state.append(dst_data)

            # 删除self.dst_state_path文件夹
            if os.path.exists(self.dst_state_path):
                shutil.rmtree(self.dst_state_path)

            # 删除self.src_state_path文件夹
            if os.path.exists(self.src_state_path):
                shutil.rmtree(self.src_state_path)


Model = MEModel
