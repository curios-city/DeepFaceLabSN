import multiprocessing
import shutil

from DFLIMG import *
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from core import pathex
from core.cv2ex import *


class FacesetEnhancerSubprocessor(Subprocessor):

    #override
    def __init__(self, image_paths, output_dirpath, device_config):
        self.image_paths = image_paths
        self.output_dirpath = output_dirpath
        self.result = []
        self.nn_initialize_mp_lock = multiprocessing.Lock()
        self.devices = FacesetEnhancerSubprocessor.get_devices_for_config(device_config)

        super().__init__('FacesetEnhancer', FacesetEnhancerSubprocessor.Cli, 600)

    #override
    def on_clients_initialized(self):
        io.progress_bar (None, len (self.image_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        base_dict = {'output_dirpath':self.output_dirpath,
                     'nn_initialize_mp_lock': self.nn_initialize_mp_lock,}

        for (device_idx, device_type, device_name, device_total_vram_gb) in self.devices:
            client_dict = base_dict.copy()
            client_dict['device_idx'] = device_idx
            client_dict['device_name'] = device_name
            client_dict['device_type'] = device_type
            yield client_dict['device_name'], {}, client_dict

    #override
    def get_data(self, host_dict):
        if len (self.image_paths) > 0:
            return self.image_paths.pop(0)

    #override
    def on_data_return (self, host_dict, data):
        self.image_paths.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        io.progress_bar_inc(1)
        if result[0] == 1:
            self.result +=[ (result[1], result[2]) ]

    #override
    def get_result(self):
        return self.result

    @staticmethod
    def get_devices_for_config (device_config):
        devices = device_config.devices
        cpu_only = len(devices) == 0

        if not cpu_only:
            return [ (device.index, 'GPU', device.name, device.total_mem_gb) for device in devices ]
        else:
            return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in range( min(8, multiprocessing.cpu_count() // 2) ) ]

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            device_idx   = client_dict['device_idx']
            cpu_only     = client_dict['device_type'] == 'CPU'
            self.output_dirpath = client_dict['output_dirpath']
            nn_initialize_mp_lock = client_dict['nn_initialize_mp_lock']

            if cpu_only:
                device_config = nn.DeviceConfig.CPU()
                device_vram = 99
            else:
                device_config = nn.DeviceConfig.GPUIndexes ([device_idx])
                device_vram = device_config.devices[0].total_mem_gb

            nn.initialize (device_config)

            intro_str = 'Running on %s.' % (client_dict['device_name'])

            self.log_info (intro_str)

            from facelib import FaceEnhancer
            self.fe = FaceEnhancer( place_model_on_cpu=(device_vram<=2 or cpu_only), run_on_cpu=cpu_only )

        #override
        def process_data(self, filepath):
            try:
                dflimg = DFLIMG.load (filepath)
                if dflimg is None or not dflimg.has_data():
                    self.log_err (f"{filepath.name} is not a dfl image file")
                else:
                    dfl_dict = dflimg.get_dict()

                    img = cv2_imread(filepath).astype(np.float32) / 255.0
                    img = self.fe.enhance(img)
                    img = np.clip (img*255, 0, 255).astype(np.uint8)

                    output_filepath = self.output_dirpath / filepath.name

                    cv2_imwrite ( str(output_filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )

                    dflimg = DFLIMG.load (output_filepath)
                    dflimg.set_dict(dfl_dict)
                    dflimg.save()

                    return (1, filepath, output_filepath)
            except:
                self.log_err (f"Exception occured while processing file {filepath}. Error: {traceback.format_exc()}")

            return (0, filepath, None)

def process_folder ( dirpath, cpu_only=False, force_gpu_idxs=None ):
    device_config = nn.DeviceConfig.GPUIndexes( force_gpu_idxs or nn.ask_choose_device_idxs(suggest_all_gpu=True) ) \
                    if not cpu_only else nn.DeviceConfig.CPU()

    output_dirpath = dirpath.parent / (dirpath.name + '_enhanced')
    output_dirpath.mkdir (exist_ok=True, parents=True)

    dirpath_parts = '/'.join( dirpath.parts[-2:])
    output_dirpath_parts = '/'.join( output_dirpath.parts[-2:] )
    io.log_info (f"Enhancing faceset in {dirpath_parts}")
    io.log_info ( f"Processing to {output_dirpath_parts}")

    output_images_paths = pathex.get_image_paths(output_dirpath)
    if len(output_images_paths) > 0:
        for filename in output_images_paths:
            Path(filename).unlink()

    image_paths = [Path(x) for x in pathex.get_image_paths( dirpath )]
    result = FacesetEnhancerSubprocessor ( image_paths, output_dirpath, device_config=device_config).run()

    is_merge = io.input_bool (f"\r\nMerge {output_dirpath_parts} to {dirpath_parts} ?", True)
    if is_merge:
        io.log_info (f"Copying processed files to {dirpath_parts}")

        for (filepath, output_filepath) in result:
            try:
                shutil.copy (output_filepath, filepath)
            except:
                pass

        io.log_info (f"Removing {output_dirpath_parts}")
        shutil.rmtree(output_dirpath)
