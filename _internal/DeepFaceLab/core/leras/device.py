import sys
import ctypes
import os
import multiprocessing
import json
import time
from pathlib import Path
from core.interact import interact as io


class Device(object):
    def __init__(self, index, tf_dev_type, name, total_mem, free_mem):
        self.index = index
        self.tf_dev_type = tf_dev_type
        self.name = name
        
        self.total_mem = total_mem
        self.total_mem_gb = total_mem / 1024**3
        self.free_mem = free_mem
        self.free_mem_gb = free_mem / 1024**3

    def __str__(self):
        return f"[{self.index}]:[{self.name}][{self.free_mem_gb:.3}/{self.total_mem_gb :.3}]"

class Devices(object):
    all_devices = None

    def __init__(self, devices):
        self.devices = devices

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, key):
        result = self.devices[key]
        if isinstance(key, slice):
            return Devices(result)
        return result

    def __iter__(self):
        for device in self.devices:
            yield device

    def get_best_device(self):
        result = None
        idx_mem = 0
        for device in self.devices:
            mem = device.total_mem
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_worst_device(self):
        result = None
        idx_mem = sys.maxsize
        for device in self.devices:
            mem = device.total_mem
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_device_by_index(self, idx):
        for device in self.devices:
            if device.index == idx:
                return device
        return None

    def get_devices_from_index_list(self, idx_list):
        result = []
        for device in self.devices:
            if device.index in idx_list:
                result += [device]
        return Devices(result)

    def get_equal_devices(self, device):
        device_name = device.name
        result = []
        for device in self.devices:
            if device.name == device_name:
                result.append (device)
        return Devices(result)

    def get_devices_at_least_mem(self, totalmemsize_gb):
        result = []
        for device in self.devices:
            if device.total_mem >= totalmemsize_gb*(1024**3):
                result.append (device)
        return Devices(result)

    @staticmethod
    def _get_tf_devices_proc(q : multiprocessing.Queue):
        
        if sys.platform[0:3] == 'win':
            compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('ComputeCache_ALL')
            os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
            if not compute_cache_path.exists():
                io.log_info("缓存 GPU 内核..")
                compute_cache_path.mkdir(parents=True, exist_ok=True)
                
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

        from tensorflow.python.client import device_lib

        devices = []
        
        physical_devices = device_lib.list_local_devices()
        physical_devices_f = {}
        for dev in physical_devices:
            dev_type = dev.device_type
            dev_tf_name = dev.name
            dev_tf_name = dev_tf_name[ dev_tf_name.index(dev_type) : ]
            
            dev_idx = int(dev_tf_name.split(':')[-1])
            
            if dev_type in ['GPU','DML']:
                dev_name = dev_tf_name
                
                dev_desc = dev.physical_device_desc
                if len(dev_desc) != 0:
                    if dev_desc[0] == '{':
                        dev_desc_json = json.loads(dev_desc)
                        dev_desc_json_name = dev_desc_json.get('name',None)
                        if dev_desc_json_name is not None:
                            dev_name = dev_desc_json_name
                    else:
                        for param, value in ( v.split(':') for v in dev_desc.split(',') ):
                            param = param.strip()
                            value = value.strip()
                            if param == 'name':
                                dev_name = value
                                break
                
                physical_devices_f[dev_idx] = (dev_type, dev_name, dev.memory_limit)
                        
        q.put(physical_devices_f)
        time.sleep(0.1)
        
        
    @staticmethod
    def initialize_main_env():
        if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 0:
            return
            
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        
        os.environ['TF_DIRECTML_KERNEL_CACHE_SIZE'] = '2500'
        os.environ['CUDA_​CACHE_​MAXSIZE'] = '2147483647'
        os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tf log errors only
        
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=Devices._get_tf_devices_proc, args=(q,), daemon=True)
        p.start()
        p.join()
        
        visible_devices = q.get()

        os.environ['NN_DEVICES_INITIALIZED'] = '1'
        os.environ['NN_DEVICES_COUNT'] = str(len(visible_devices))
        
        for i in visible_devices:
            dev_type, name, total_mem = visible_devices[i]

            os.environ[f'NN_DEVICE_{i}_TF_DEV_TYPE'] = dev_type
            os.environ[f'NN_DEVICE_{i}_NAME'] = name
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(total_mem)
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(total_mem)
            
        

    @staticmethod
    def getDevices():
        if Devices.all_devices is None:
            if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 1:
                raise Exception("nn devices are not initialized. Run initialize_main_env() in main process.")
            devices = []
            for i in range ( int(os.environ['NN_DEVICES_COUNT']) ):
                devices.append ( Device(index=i,
                                        tf_dev_type=os.environ[f'NN_DEVICE_{i}_TF_DEV_TYPE'],
                                        name=os.environ[f'NN_DEVICE_{i}_NAME'],
                                        total_mem=int(os.environ[f'NN_DEVICE_{i}_TOTAL_MEM']),
                                        free_mem=int(os.environ[f'NN_DEVICE_{i}_FREE_MEM']), )
                                )
            Devices.all_devices = Devices(devices)

        return Devices.all_devices

"""

        
        # {'name'      : name.split(b'\0', 1)[0].decode(),
        #     'total_mem' : totalMem.value
        # }

        
        
        
        
        return

        
        
        
        min_cc = int(os.environ.get("TF_MIN_REQ_CAP", 35))
        libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll')
        for libname in libnames:
            try:
                cuda = ctypes.CDLL(libname)
            except:
                continue
            else:
                break
        else:
            return Devices([])

        nGpus = ctypes.c_int()
        name = b' ' * 200
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        freeMem = ctypes.c_size_t()
        totalMem = ctypes.c_size_t()

        result = ctypes.c_int()
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        error_str = ctypes.c_char_p()

        devices = []

        if cuda.cuInit(0) == 0 and \
            cuda.cuDeviceGetCount(ctypes.byref(nGpus)) == 0:
            for i in range(nGpus.value):
                if cuda.cuDeviceGet(ctypes.byref(device), i) != 0 or \
                    cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) != 0 or \
                    cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != 0:
                    continue

                if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device) == 0:
                    if cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem)) == 0:
                        cc = cc_major.value * 10 + cc_minor.value
                        if cc >= min_cc:
                            devices.append ( {'name'      : name.split(b'\0', 1)[0].decode(),
                                              'total_mem' : totalMem.value,
                                              'free_mem'  : freeMem.value,
                                              'cc'        : cc
                                              })
                    cuda.cuCtxDetach(context)

        os.environ['NN_DEVICES_COUNT'] = str(len(devices))
        for i, device in enumerate(devices):
            os.environ[f'NN_DEVICE_{i}_NAME'] = device['name']
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(device['total_mem'])
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(device['free_mem'])
            os.environ[f'NN_DEVICE_{i}_CC'] = str(device['cc'])
"""
