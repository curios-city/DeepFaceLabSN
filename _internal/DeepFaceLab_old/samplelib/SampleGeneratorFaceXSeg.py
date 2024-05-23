import multiprocessing  # 导入multiprocessing模块，用于多进程处理
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import time  # 导入time模块，用于处理时间相关的操作
import traceback  # 导入traceback模块，用于打印异常堆栈信息
from enum import IntEnum  # 导入IntEnum类，用于创建枚举类型

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
from pathlib import Path  # 导入Path类，用于处理文件路径
from core import imagelib, mplib, pathex  # 导入自定义模块
from core.imagelib import sd  # 从imagelib模块导入sd函数
from core.cv2ex import *  # 导入core.cv2ex模块中的所有内容
from core.interact import interact as io  # 导入core.interact模块中的interact函数并重命名为io
from core.joblib import Subprocessor, SubprocessGenerator, ThisThreadGenerator  # 导入自定义模块中的类
from facelib import LandmarksProcessor  # 导入facelib模块中的LandmarksProcessor类
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor, SampleType)  # 导入samplelib模块中的类和枚举

# 定义一个名为SampleGeneratorFaceXSeg的类，继承自SampleGeneratorBase类
class SampleGeneratorFaceXSeg(SampleGeneratorBase):
    # 定义初始化方法，接收一系列参数
    def __init__ (self, paths, debug=False, batch_size=1, resolution=256, face_type=None,
                        generators_count=4, data_format="NHWC",
                        **kwargs):
        super().__init__(debug, batch_size)  # 调用父类的初始化方法

        self.initialized = False  # 初始化一个标志位，表示是否已经完成初始化

        # 将所有路径下的面部样本加载到samples列表中
        samples = sum([SampleLoader.load(SampleType.FACE, path) for path in paths])
        # 使用SegmentedSampleFilterSubprocessor处理样本，获取分割后的样本索引
        seg_sample_idxs = SegmentedSampleFilterSubprocessor(samples).run()

        # 如果没有找到分割后的样本
        if len(seg_sample_idxs) == 0:
            # 使用SegmentedSampleFilterSubprocessor重新处理样本，并计算xseg_mask的数量
            seg_sample_idxs = SegmentedSampleFilterSubprocessor(samples, count_xseg_mask=True).run()
            # 如果还是没有找到分割后的样本
            if len(seg_sample_idxs) == 0:
                # 抛出异常
                raise Exception(f"未发现 已写遮罩 的图片.")
            else:
                # 打印信息，表示使用了xseg标记的样本
                io.log_info(f"使用 {len(seg_sample_idxs)} 张 已写遮罩 的图片.")
        else:
            # 打印信息，表示使用了分割后的样本
            self.seg_sample_count = len(seg_sample_idxs)
            io.log_info(f"使用 {len(seg_sample_idxs)} 张 手动绘制.")

        # 如果处于调试模式
        if self.debug:
            self.generators_count = 1  # 设置生成器数量为1
        else:
            # 否则，将生成器数量设置为指定数量和1之间的较大值
            self.generators_count = max(1, generators_count)

        args = (samples, seg_sample_idxs, resolution, face_type, data_format)  # 定义生成器参数元组

        # 如果处于调试模式
        if self.debug:
            # 创建一个运行在当前线程中的生成器
            self.generators = [ThisThreadGenerator(self.batch_func, args)]
        else:
            # 否则，创建指定数量的子进程生成器，并不立即启动
            self.generators = [SubprocessGenerator(self.batch_func, args, start_now=False) for i in range(self.generators_count)]

            # 启动所有子进程生成器
            SubprocessGenerator.start_in_parallel(self.generators)

        self.generator_counter = -1  # 初始化生成器计数器为-1，用于轮询生成器

        self.initialized = True  # 完成初始化标志位设置为True

    # 定义一个可重写的方法，用于检查是否已经完成初始化
    def is_initialized(self):
        return self.initialized

    # 定义一个方法，使得类的实例可以迭代
    def __iter__(self):
        return self

    # 定义一个方法，用于获取下一个批次的样本数据
    def __next__(self):
        self.generator_counter += 1  # 递增生成器计数器
        generator = self.generators[self.generator_counter % len(self.generators)]  # 获取当前生成器
        return next(generator)  # 返回当前生成器的下一个样本

    # 定义一个方法，用于生成批次数据
    def batch_func(self, param):
        samples, seg_sample_idxs, resolution, face_type, data_format = param  # 解析参数

        shuffle_idxs = []  # 初始化乱序索引列表
        bg_shuffle_idxs = []  # 初始化背景乱序索引列表

        # 设置数据增强的参数
        random_flip = True
        rotation_range = [-8, 8]
        scale_range = [-0.1, 0.1]
        tx_range = [-0.05, 0.05]
        ty_range = [-0.05, 0.05]
        random_bilinear_resize_chance, random_bilinear_resize_max_size_per = 25, 75
        sharpen_chance, sharpen_kernel_max_size = 25, 5
        motion_blur_chance, motion_blur_mb_max_size = 25, 5
        gaussian_blur_chance, gaussian_blur_kernel_max_size = 25, 5
        random_jpeg_compress_chance = 25

        # 定义一个函数，用于生成图像和掩码
        def gen_img_mask(sample):
            img = sample.load_bgr()  # 加载BGR格式的图像
            h, w, c = img.shape  # 获取图像的高度、宽度和通道数

            # 如果样本具有分割后的多边形掩码
            if sample.seg_ie_polys.has_polys():
                mask = np.zeros((h, w, 1), dtype=np.float32)  # 创建一个全零的掩码
                sample.seg_ie_polys.overlay_mask(mask)  # 在掩码上绘制多边形
            # 如果样本具有xseg标记掩码
            elif sample.has_xseg_mask():
                mask = sample.get_xseg_mask()  # 获取xseg标记掩码
                mask[mask < 0.5] = 0.0  # 将小于0.5的像素值设为0
                mask[mask >= 0.5] = 1.0  # 将大于等于0.5的像素值设为1
            else:
                # 否则，抛出异常，表示样本中没有掩码
                raise Exception(f'no mask in sample {sample.filename}')

            # 根据人脸类型对图像和掩码进行变换
            if face_type == sample.face_type:
                # 如果图像宽度不等于指定的分辨率
                if w != resolution:
                    # 对图像和掩码进行插值缩放
                    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
                    mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
            else:
                # 否则，根据人脸关键点获取变换矩阵，对图像和掩码进行仿射变换
                mat = LandmarksProcessor.get_transform_mat(sample.landmarks, resolution, face_type)
                img = cv2.warpAffine(img, mat, (resolution, resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4)
                mask = cv2.warpAffine(mask, mat, (resolution, resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4)

            if len(mask.shape) == 2:
                mask = mask[..., None]  # 如果掩码的维度为2，则在末尾添加一个维度
            return img, mask  # 返回处理后的图像和掩码


        bs = self.batch_size  # 设置批处理大小

        while True:  # 无限循环，确保每次迭代都会生成一个新的批次
            batches = [[], []]  # 初始化批次列表，用于存储图像和掩码数据

            n_batch = 0  # 初始化批次计数器
            while n_batch < bs:  # 当批次计数器小于批处理大小时执行以下操作
                try:
                    if len(shuffle_idxs) == 0:  # 如果shuffle_idxs列表为空，则重新填充
                        shuffle_idxs = seg_sample_idxs.copy()  # 复制seg_sample_idxs列表
                        np.random.shuffle(shuffle_idxs)  # 对shuffle_idxs列表进行随机重排
                    sample = samples[shuffle_idxs.pop()]  # 从samples中获取样本，pop()方法弹出并返回shuffle_idxs中的最后一个元素

                    # 生成图像和掩码
                    img, mask = gen_img_mask(sample)

                    # 以50%的概率执行以下操作
                    if np.random.randint(2) == 0:
                        if len(bg_shuffle_idxs) == 0:
                            bg_shuffle_idxs = seg_sample_idxs.copy()
                            np.random.shuffle(bg_shuffle_idxs)
                        bg_sample = samples[bg_shuffle_idxs.pop()]

                        bg_img, bg_mask = gen_img_mask(bg_sample)

                        # 生成背景图像的变换参数
                        bg_wp = imagelib.gen_warp_params(resolution, True, rotation_range=[-180, 180], scale_range=[-0.10, 0.10], tx_range=[-0.10, 0.10], ty_range=[-0.10, 0.10])
                        bg_img = imagelib.warp_by_params(bg_wp, bg_img, can_warp=False, can_transform=True, can_flip=True, border_replicate=True)
                        bg_mask = imagelib.warp_by_params(bg_wp, bg_mask, can_warp=False, can_transform=True, can_flip=True, border_replicate=False)
                        bg_img = bg_img * (1 - bg_mask)
                        if np.random.randint(2) == 0:
                            bg_img = imagelib.apply_random_hsv_shift(bg_img)
                        else:
                            bg_img = imagelib.apply_random_rgb_levels(bg_img)

                        c_mask = 1.0 - (1 - bg_mask) * (1 - mask)
                        rnd = 0.15 + np.random.uniform() * 0.85
                        img = img * (c_mask) + img * (1 - c_mask) * rnd + bg_img * (1 - c_mask) * (1 - rnd)

                    # 生成图像的变换参数
                    warp_params = imagelib.gen_warp_params(resolution, random_flip, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range)
                    img = imagelib.warp_by_params(warp_params, img, can_warp=True, can_transform=True, can_flip=True, border_replicate=True)
                    mask = imagelib.warp_by_params(warp_params, mask, can_warp=True, can_transform=True, can_flip=True, border_replicate=False)

                    img = np.clip(img.astype(np.float32), 0, 1)  # 将图像像素值剪切到0和1之间
                    mask[mask < 0.5] = 0.0  # 将掩码中小于0.5的像素值设为0
                    mask[mask >= 0.5] = 1.0  # 将掩码中大于等于0.5的像素值设为1
                    mask = np.clip(mask, 0, 1)  # 将掩码像素值剪切到0和1之间

                    # 以50%的概率执行以下操作
                    if np.random.randint(2) == 0:
                        # 随机添加人脸光斑
                        krn = np.random.randint(resolution // 4, resolution)
                        krn = krn - krn % 2 + 1
                        img = img + cv2.GaussianBlur(img * mask, (krn, krn), 0)

                    # 以50%的概率执行以下操作
                    if np.random.randint(2) == 0:
                        # 随机添加背景光斑
                        krn = np.random.randint(resolution // 4, resolution)
                        krn = krn - krn % 2 + 1
                        img = img + cv2.GaussianBlur(img * (1 - mask), (krn, krn), 0)

                    # 以50%的概率执行以下操作
                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_hsv_shift(img, mask=sd.random_circle_faded([resolution, resolution]))
                    else:
                        img = imagelib.apply_random_rgb_levels(img, mask=sd.random_circle_faded([resolution, resolution]))

                    # 以50%的概率执行以下操作
                    if np.random.randint(2) == 0:
                        # 随机应用锐化操作
                        img = imagelib.apply_random_sharpen(img, sharpen_chance, sharpen_kernel_max_size, mask=sd.random_circle_faded([resolution, resolution]))
                    else:
                        # 随机应用运动模糊和高斯模糊操作
                        img = imagelib.apply_random_motion_blur(img, motion_blur_chance, motion_blur_mb_max_size, mask=sd.random_circle_faded([resolution, resolution]))
                        img = imagelib.apply_random_gaussian_blur(img, gaussian_blur_chance, gaussian_blur_kernel_max_size, mask=sd.random_circle_faded([resolution, resolution]))

                    # 以50%的概率执行以下操作
                    if np.random.randint(2) == 0:
                        # 随机应用最近邻插值调整图像大小
                        img = imagelib.apply_random_nearest_resize(img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded([resolution, resolution]))
                    else:
                        # 随机应用双线性插值调整图像大小
                        img = imagelib.apply_random_bilinear_resize(img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded([resolution, resolution]))

                    img = np.clip(img, 0, 1)  # 将图像像素值剪切到0和1之间

                    # 随机应用JPEG压缩
                    img = imagelib.apply_random_jpeg_compress(img, random_jpeg_compress_chance, mask=sd.random_circle_faded([resolution, resolution]))

                    if data_format == "NCHW":  # 如果数据格式为"NCHW"
                        img = np.transpose(img, (2,0,1))  # 转置图像维度顺序为通道高度宽度
                        mask = np.transpose(mask, (2,0,1))  # 转置掩码维度顺序为通道高度宽度

                    batches[0].append(img)  # 将处理后的图像添加到批次的第一个列表中
                    batches[1].append(mask)  # 将处理后的掩码添加到批次的第二个列表中


                    n_batch += 1
                except:
                    io.log_err ( traceback.format_exc() )

            yield [ np.array(batch) for batch in batches]

class SegmentedSampleFilterSubprocessor(Subprocessor):
    #override
    def __init__(self, samples, count_xseg_mask=False ):
        self.samples = samples  # 存储样本集合
        self.samples_len = len(self.samples)  # 获取样本集合的长度
        self.count_xseg_mask = count_xseg_mask  # 是否计算xseg_mask的标志

        self.idxs = [*range(self.samples_len)]  # 初始化索引列表
        self.result = []  # 存储处理结果的列表
        super().__init__('SegmentedSampleFilterSubprocessor', SegmentedSampleFilterSubprocessor.Cli, 60)  # 调用父类的初始化方法

    #override
    def process_info_generator(self):
        for i in range(multiprocessing.cpu_count()):  # 遍历CPU的数量
            yield 'CPU%d' % (i), {}, {'samples':self.samples, 'count_xseg_mask':self.count_xseg_mask}  # 生成处理信息

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Filtering", self.samples_len)  # 显示过滤进度条

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()  # 关闭进度条

    #override
    def get_data(self, host_dict):
        if len (self.idxs) > 0:
            return self.idxs.pop(0)  # 弹出索引列表的第一个元素作为数据返回
        return None  # 如果索引列表为空，则返回None

    #override
    def on_data_return (self, host_dict, data):
        self.idxs.insert(0, data)  # 将数据插入到索引列表的开头

    #override
    def on_result (self, host_dict, data, result):
        idx, is_ok = result  # 解包处理结果
        if is_ok:  # 如果处理成功
            self.result.append(idx)  # 将索引添加到结果列表中
        io.progress_bar_inc(1)  # 增加进度条的进度

    def get_result(self):
        return self.result  # 返回处理结果列表

    class Cli(Subprocessor.Cli):
        #overridable optional
        def on_initialize(self, client_dict):
            self.samples = client_dict['samples']  # 从client_dict中获取样本集合
            self.count_xseg_mask = client_dict['count_xseg_mask']  # 从client_dict中获取计算xseg_mask的标志

        def process_data(self, idx):
            if self.count_xseg_mask:  # 如果需要计算xseg_mask
                return idx, self.samples[idx].has_xseg_mask()  # 返回索引和样本是否具有xseg_mask
            else:
                return idx, self.samples[idx].seg_ie_polys.get_pts_count() != 0  # 返回索引和样本seg_ie_polys的点数是否不为0


"""
  bg_path = None
        for path in paths:
            bg_path = Path(path) / 'backgrounds'
            if bg_path.exists():

                break
        if bg_path is None:
            io.log_info(f'Random backgrounds will not be used. Place no face jpg images to aligned\backgrounds folder. ')
            bg_pathes = None
        else:
            bg_pathes = pathex.get_image_paths(bg_path, image_extensions=['.jpg'], return_Path_class=True)
            io.log_info(f'Using {len(bg_pathes)} random backgrounds from {bg_path}')

if bg_pathes is not None:
            bg_path = bg_pathes[ np.random.randint(len(bg_pathes)) ]

            bg_img = cv2_imread(bg_path)
            if bg_img is not None:
                bg_img = bg_img.astype(np.float32) / 255.0
                bg_img = imagelib.normalize_channels(bg_img, 3)

                bg_img = imagelib.random_crop(bg_img, resolution, resolution)
                bg_img = cv2.resize(bg_img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

            if np.random.randint(2) == 0:
                bg_img = imagelib.apply_random_hsv_shift(bg_img)
            else:
                bg_img = imagelib.apply_random_rgb_levels(bg_img)

            bg_wp   = imagelib.gen_warp_params(resolution, True, rotation_range=[-180,180], scale_range=[0,0], tx_range=[0,0], ty_range=[0,0])
            bg_img  = imagelib.warp_by_params (bg_wp, bg_img,  can_warp=False, can_transform=True, can_flip=True, border_replicate=True)

            bg = img*(1-mask)
            fg = img*mask

            c_mask = sd.random_circle_faded ([resolution,resolution])
            bg = ( bg_img*c_mask + bg*(1-c_mask) )*(1-mask)

            img = fg+bg

        else:
"""