"""
Leras。
像轻量级的 Keras 一样。
这是我从零开始纯用 TensorFlow 写的轻量级神经网络库，
没有使用 Keras。
提供：

1.完全自由的 TensorFlow 操作，没有 Keras 模型的限制
2.类似 PyTorch 的简单模型操作，但是在图模式下（没有即时执行）
3.方便和易懂的逻辑

不能在这里直接导入 TensorFlow 或任何 tensorflow.sub 模块的原因：
1) 程序在导入 TensorFlow 之前会根据 DeviceConfig 改变环境变量
2) 多进程将在每次生成时导入 TensorFlow
NCHW 可以加速训练速度 10-20%。
"""

import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import numpy as np
from core.interact import interact as io  # 导入交互模块
from .device import Devices  # 导入设备模块


class nn():  # 定义神经网络类

    current_DeviceConfig = None  # 当前设备配置，默认为空

    tf = None  # TensorFlow 对象，默认为空
    tf_sess = None  # TensorFlow 会话，默认为空
    tf_sess_config = None  # TensorFlow 会话配置，默认为空
    tf_default_device_name = None  # TensorFlow 默认设备名称，默认为空
    
    data_format = None  # 数据格式，默认为空
    conv2d_ch_axis = None  # 卷积层通道轴，默认为空
    conv2d_spatial_axes = None  # 卷积层空间轴，默认为空

    floatx = None  # 浮点数类型，默认为空
    
    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NHWC"):  # 初始化方法

        if nn.tf is None:  # 如果 TensorFlow 对象为空
            if device_config is None:  # 如果设备配置为空
                device_config = nn.getCurrentDeviceConfig()  # 获取当前设备配置
            nn.setCurrentDeviceConfig(device_config)  # 设置当前设备配置

            # 在导入 TensorFlow 之前操作环境变量

            first_run = False  # 是否首次运行，默认为 False
            if len(device_config.devices) != 0:  # 如果设备列表不为空
                if sys.platform[0:3] == 'win':  # 如果是 Windows 系统
                    # Windows 特定环境变量
                    if all([x.name == device_config.devices[0].name for x in device_config.devices]):
                        devices_str = "_" + device_config.devices[0].name.replace(' ', '_')
                    else:
                        devices_str = ""
                        for device in device_config.devices:
                            devices_str += "_" + device.name.replace(' ', '_')

                    compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('ComputeCache' + devices_str)
                    if not compute_cache_path.exists():
                        first_run = True
                        compute_cache_path.mkdir(parents=True, exist_ok=True)
                    os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
            
            if first_run:
                io.log_info("缓存 GPU 内核...")  # 输出信息：缓存 GPU 内核



            import tensorflow

            # 获取 TensorFlow 版本
            tf_version = tensorflow.version.VERSION
            #if tf_version is None:
            #    tf_version = tensorflow.version.GIT_VERSION
            if tf_version[0] == 'v':
                tf_version = tf_version[1:]

            # 根据 TensorFlow 的版本选择相应的导入方式
            if tf_version[0] == '2':
                tf = tensorflow.compat.v1
            else:
                tf = tensorflow

            import logging
            # 禁止 TensorFlow 的警告信息
            tf_logger = logging.getLogger('tensorflow')
            tf_logger.setLevel(logging.ERROR)

            # 根据 TensorFlow 的版本禁用相应的功能
            if tf_version[0] == '2':
                tf.disable_v2_behavior()
            nn.tf = tf

            # 初始化框架
            import core.leras.ops
            import core.leras.layers
            import core.leras.initializers
            import core.leras.optimizers
            import core.leras.models
            import core.leras.archis

            # 配置 TensorFlow 会话
            if len(device_config.devices) == 0:
                # 如果没有 GPU 设备，则将默认设备设置为 CPU
                config = tf.ConfigProto(device_count={'GPU': 0})
                nn.tf_default_device_name = '/CPU:0'
            else:
                # 如果有 GPU 设备，则根据设备配置选择第一个 GPU 设备
                nn.tf_default_device_name = f'/{device_config.devices[0].tf_dev_type}:0'
    
                config = tf.ConfigProto()
                # 设置可见的 GPU 设备列表
                config.gpu_options.visible_device_list = ','.join([str(device.index) for device in device_config.devices])
    
            # 设置 GPU 选项
            config.gpu_options.force_gpu_compatible = True
            config.gpu_options.allow_growth = True
            nn.tf_sess_config = config

            
        if nn.tf_sess is None:
            # 如果 TensorFlow 会话不存在，则创建一个新的会话
            nn.tf_sess = tf.Session(config=nn.tf_sess_config)

        if floatx == "float32":
            # 如果浮点数类型为 float32，则使用 TensorFlow 的 float32 类型
            floatx = nn.tf.float32
        elif floatx == "float16":
            # 如果浮点数类型为 float16，则使用 TensorFlow 的 float16 类型
            floatx = nn.tf.float16
        else:
            # 如果浮点数类型不支持，则引发 ValueError 错误
            raise ValueError(f"不支持的浮点数类型 {floatx}")
        nn.set_floatx(floatx)
        nn.set_data_format(data_format)

    @staticmethod
    def initialize_main_env():
        # 初始化主要环境
        Devices.initialize_main_env()

    @staticmethod
    def set_floatx(tf_dtype):
        """
        设置所有层的默认浮点数类型，当它们的 dtype 为 None 时
        """
        nn.floatx = tf_dtype

    @staticmethod
    def set_data_format(data_format):
        if data_format != "NHWC" and data_format != "NCHW":
            # 如果数据格式不支持，则引发 ValueError 错误
            raise ValueError(f"不支持的数据格式 {data_format}")
        nn.data_format = data_format

        if data_format == "NHWC":
            # 如果数据格式为 NHWC，则设置相应的通道轴和空间轴
            nn.conv2d_ch_axis = 3
            nn.conv2d_spatial_axes = [1,2]
        elif data_format == "NCHW":
            # 如果数据格式为 NCHW，则设置相应的通道轴和空间轴
            nn.conv2d_ch_axis = 1
            nn.conv2d_spatial_axes = [2,3]

    @staticmethod
    def get4Dshape(w, h, c):
        """
        根据当前数据格式返回 4D 形状
        """
        if nn.data_format == "NHWC":
            return (None, h, w, c)
        else:
            return (None, c, h, w)

    @staticmethod
    def to_data_format(x, to_data_format, from_data_format):
        """
        将输入张量 x 从当前数据格式转换为指定的数据格式

        Args:
            x: 输入张量
            to_data_format (str): 目标数据格式，支持 "NHWC" 或 "NCHW"
            from_data_format (str): 当前数据格式，支持 "NHWC" 或 "NCHW"

        Returns:
            转换后的张量

        Raises:
            ValueError: 如果指定的目标数据格式不支持

        """
        if to_data_format == from_data_format:
            return x

        if to_data_format == "NHWC":
            # 将数据从 NCHW 转换为 NHWC
            return np.transpose(x, (0, 2, 3, 1))
        elif to_data_format == "NCHW":
            # 将数据从 NHWC 转换为 NCHW
            return np.transpose(x, (0, 3, 1, 2))
        else:
            raise ValueError(f"不支持的目标数据格式 {to_data_format}")

    @staticmethod
    def getCurrentDeviceConfig():
        """
        获取当前设备配置

        Returns:
            当前设备配置

        """
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        """
        设置当前设备配置

        Args:
            device_config: 要设置的设备配置

        """
        nn.current_DeviceConfig = device_config

    @staticmethod
    def reset_session():
        """
        重置 TensorFlow 会话

        """
        if nn.tf is not None:
            if nn.tf_sess is not None:
                nn.tf.reset_default_graph()
                nn.tf_sess.close()
                nn.tf_sess = nn.tf.Session(config=nn.tf_sess_config)

    @staticmethod
    def close_session():
        """
        关闭 TensorFlow 会话

        """
        if nn.tf_sess is not None:
            nn.tf.reset_default_graph()
            nn.tf_sess.close()
            nn.tf_sess = None


    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
        """
        询问用户选择设备索引的方法。

        Args:
            choose_only_one (bool): 是否只能选择一个设备索引，默认为 False。
            allow_cpu (bool): 是否允许选择 CPU，默认为 True。
            suggest_best_multi_gpu (bool): 是否建议选择最佳的多 GPU 设备，默认为 False。
            suggest_all_gpu (bool): 是否建议选择所有 GPU 设备，默认为 False。

        Returns:
            list: 用户选择的设备索引列表。

        """
        devices = Devices.getDevices()
        if len(devices) == 0:
            return []

        all_devices_indexes = [device.index for device in devices]

        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_gpu = False

        if suggest_all_gpu:
            best_device_indexes = all_devices_indexes
        elif suggest_best_multi_gpu:
            best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device())]
        else:
            best_device_indexes = [devices.get_best_device().index]
        best_device_indexes = ",".join([str(x) for x in best_device_indexes])

        io.log_info("")
        if choose_only_one:
            io.log_info("选择一个 GPU 索引。")
        else:
            io.log_info("选择一个或多个 GPU 索引（用逗号分隔）。提示：参数较低的模型，多卡交火可能比单卡慢！")
        io.log_info("")

        if allow_cpu:
            io.log_info("[CPU] : CPU")
        for device in devices:
            io.log_info(f"  [{device.index}] : {device.name}")

        io.log_info("")

        while True:
            try:
                if choose_only_one:
                    choosed_idxs = io.input_str("请选择 GPU 索引？", best_device_indexes)
                else:
                    choosed_idxs = io.input_str("请选择 GPU 索引？", best_device_indexes)

                if allow_cpu and choosed_idxs.lower() == "cpu":
                    choosed_idxs = []
                    break

                choosed_idxs = [int(x) for x in choosed_idxs.split(',')]

                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break
                else:
                    if all([idx in all_devices_indexes for idx in choosed_idxs]):
                        break
            except:
                pass
        io.log_info("")

        return choosed_idxs


    class DeviceConfig():
        @staticmethod
        def ask_choose_device(*args, **kwargs):
            """
            静态方法：询问用户选择设备。

            Args:
                *args: 传递给 ask_choose_device_idxs() 方法的位置参数。
                **kwargs: 传递给 ask_choose_device_idxs() 方法的关键字参数。

            Returns:
                DeviceConfig: 包含用户选择的设备索引的 DeviceConfig 对象。
            """
            return nn.DeviceConfig.GPUIndexes(nn.ask_choose_device_idxs(*args, **kwargs))

        def __init__(self, devices=None):
            """
            初始化 DeviceConfig 类。

            Args:
                devices (list): 设备列表，默认为 None。
            """
            devices = devices or []

            if not isinstance(devices, Devices):
                devices = Devices(devices)

            self.devices = devices
            self.cpu_only = len(devices) == 0

        @staticmethod
        def BestGPU():
            """
            静态方法：获取最佳 GPU 设备。

            Returns:
                DeviceConfig: 包含最佳 GPU 设备的 DeviceConfig 对象。
            """
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_best_device()])

        @staticmethod
        def WorstGPU():
            """
            静态方法：获取最差 GPU 设备。

            Returns:
                DeviceConfig: 包含最差 GPU 设备的 DeviceConfig 对象。
            """
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_worst_device()])

        @staticmethod
        def GPUIndexes(indexes):
            """
            静态方法：根据索引列表创建 DeviceConfig 对象。

            Args:
                indexes (list): 设备索引列表。

            Returns:
                DeviceConfig: 包含指定设备索引的 DeviceConfig 对象。
            """
            if len(indexes) != 0:
                devices = Devices.getDevices().get_devices_from_index_list(indexes)
            else:
                devices = []

            return nn.DeviceConfig(devices)

        @staticmethod
        def CPU():
            """
            静态方法：获取 CPU 设备的 DeviceConfig 对象。

            Returns:
                DeviceConfig: 包含 CPU 设备的 DeviceConfig 对象。
            """
            return nn.DeviceConfig([])

