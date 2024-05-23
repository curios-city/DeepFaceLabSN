"""
Leras.

like lighter keras.
This is my lightweight neural network library written from scratch
based on pure tensorflow without keras.

Provides:
+ full freedom of tensorflow operations without keras model's restrictions
+ easy model operations like in PyTorch, but in graph mode (no eager execution)
+ convenient and understandable logic

Reasons why we cannot import tensorflow or any tensorflow.sub modules right here:
1) program is changing env variables based on DeviceConfig before import tensorflow
2) multiprocesses will import tensorflow every spawn

NCHW speed up training for 10-20%.
"""

import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import numpy as np
from core.interact import interact as io
from .device import Devices


class nn():
    current_DeviceConfig = None

    tf = None
    tf_sess = None
    tf_sess_config = None
    tf_default_device_name = None
    
    data_format = None
    conv2d_ch_axis = None
    conv2d_spatial_axes = None

    floatx = None

    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NHWC"):

        if nn.tf is None:
            if device_config is None:
                device_config = nn.getCurrentDeviceConfig()
            nn.setCurrentDeviceConfig(device_config)

            # Manipulate environment variables before import tensorflow

            first_run = False
            if len(device_config.devices) != 0:
                if sys.platform[0:3] == 'win':
                    # Windows specific env vars
                    if all( [ x.name == device_config.devices[0].name for x in device_config.devices ] ):
                        devices_str = "_" + device_config.devices[0].name.replace(' ','_')
                    else:
                        devices_str = ""
                        for device in device_config.devices:
                            devices_str += "_" + device.name.replace(' ','_')

                    compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('ComputeCache' + devices_str)
                    if not compute_cache_path.exists():
                        first_run = True
                        compute_cache_path.mkdir(parents=True, exist_ok=True)
                    os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
            
            if first_run:
                io.log_info("缓存 GPU 内核...")

            import tensorflow

            tf_version = tensorflow.version.VERSION
            #if tf_version is None:
            #    tf_version = tensorflow.version.GIT_VERSION
            if tf_version[0] == 'v':
                tf_version = tf_version[1:]
            if tf_version[0] == '2':
                tf = tensorflow.compat.v1
            else:
                tf = tensorflow

            import logging
            # Disable tensorflow warnings
            tf_logger = logging.getLogger('tensorflow')
            tf_logger.setLevel(logging.ERROR)

            if tf_version[0] == '2':
                tf.disable_v2_behavior()
            nn.tf = tf

            # Initialize framework
            import core.leras.ops
            import core.leras.layers
            import core.leras.initializers
            import core.leras.optimizers
            import core.leras.models
            import core.leras.archis

            # Configure tensorflow session-config
            if len(device_config.devices) == 0:
                config = tf.ConfigProto(device_count={'GPU': 0})
                nn.tf_default_device_name = '/CPU:0'
            else:
                nn.tf_default_device_name = f'/{device_config.devices[0].tf_dev_type}:0'
                
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.visible_device_list = ','.join([str(device.index) for device in device_config.devices])
                
            config.gpu_options.force_gpu_compatible = True
            config.gpu_options.allow_growth = True
            nn.tf_sess_config = config

        if nn.tf_sess is None:
            nn.tf_sess = tf.Session(config=nn.tf_sess_config)

        if floatx == "float32":
            floatx = nn.tf.float32
        elif floatx == "float16":
            floatx = nn.tf.float16
        else:
            raise ValueError(f"unsupported floatx {floatx}")
        nn.set_floatx(floatx)
        nn.set_data_format(data_format)

    @staticmethod
    def initialize_main_env():
        Devices.initialize_main_env()

    @staticmethod
    def set_floatx(tf_dtype):
        """
        set default float type for all layers when dtype is None for them
        """
        nn.floatx = tf_dtype

    @staticmethod
    def set_data_format(data_format):
        if data_format != "NHWC" and data_format != "NCHW":
            raise ValueError(f"unsupported data_format {data_format}")
        nn.data_format = data_format

        if data_format == "NHWC":
            nn.conv2d_ch_axis = 3
            nn.conv2d_spatial_axes = [1,2]
        elif data_format == "NCHW":
            nn.conv2d_ch_axis = 1
            nn.conv2d_spatial_axes = [2,3]

    @staticmethod
    def get4Dshape ( w, h, c ):
        """
        returns 4D shape based on current data_format
        """
        if nn.data_format == "NHWC":
            return (None,h,w,c)
        else:
            return (None,c,h,w)

    @staticmethod
    def to_data_format( x, to_data_format, from_data_format):
        if to_data_format == from_data_format:
            return x

        if to_data_format == "NHWC":
            return np.transpose(x, (0,2,3,1) )
        elif to_data_format == "NCHW":
            return np.transpose(x, (0,3,1,2) )
        else:
            raise ValueError(f"unsupported to_data_format {to_data_format}")

    @staticmethod
    def getCurrentDeviceConfig():
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        nn.current_DeviceConfig = device_config

    @staticmethod
    def reset_session():
        if nn.tf is not None:
            if nn.tf_sess is not None:
                nn.tf.reset_default_graph()
                nn.tf_sess.close()
                nn.tf_sess = nn.tf.Session(config=nn.tf_sess_config)

    @staticmethod
    def close_session():
        if nn.tf_sess is not None:
            nn.tf.reset_default_graph()
            nn.tf_sess.close()
            nn.tf_sess = None

    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
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
            best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device()) ]
        else:
            best_device_indexes = [ devices.get_best_device().index ]
        best_device_indexes = ",".join([str(x) for x in best_device_indexes])

        io.log_info ("")
        if choose_only_one:
            io.log_info ("选择一个 GPU 索引.")
        else:
            io.log_info ("选择一个或多个 GPU 索引（用逗号分隔）.")
        io.log_info ("")

        if allow_cpu:
            io.log_info ("[CPU] : CPU")
        for device in devices:
            io.log_info (f"  [{device.index}] : {device.name}")

        io.log_info ("")

        while True:
            try:
                if choose_only_one:
                    choosed_idxs = io.input_str("选择哪个 GPU 索引?", best_device_indexes)
                else:
                    choosed_idxs = io.input_str("选择哪些 GPU 索引?", best_device_indexes)

                if allow_cpu and choosed_idxs.lower() == "cpu":
                    choosed_idxs = []
                    break

                choosed_idxs = [ int(x) for x in choosed_idxs.split(',') ]

                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break
                else:
                    if all( [idx in all_devices_indexes for idx in choosed_idxs] ):
                        break
            except:
                pass
        io.log_info ("")

        return choosed_idxs

    class DeviceConfig():
        @staticmethod
        def ask_choose_device(*args, **kwargs):
            return nn.DeviceConfig.GPUIndexes( nn.ask_choose_device_idxs(*args,**kwargs) )

        def __init__ (self, devices=None):
            devices = devices or []

            if not isinstance(devices, Devices):
                devices = Devices(devices)

            self.devices = devices
            self.cpu_only = len(devices) == 0

        @staticmethod
        def BestGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_best_device()])

        @staticmethod
        def WorstGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_worst_device()])

        @staticmethod
        def GPUIndexes(indexes):
            if len(indexes) != 0:
                devices = Devices.getDevices().get_devices_from_index_list(indexes)
            else:
                devices = []

            return nn.DeviceConfig(devices)

        @staticmethod
        def CPU():
            return nn.DeviceConfig([])
