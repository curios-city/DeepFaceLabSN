import multiprocessing
import pickle
import time
import traceback
from enum import IntEnum

import cv2
import numpy as np
from pathlib import Path
from core import imagelib, mplib, pathex
from core.imagelib import sd
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor, SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor, FaceType
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor, SampleType)

class SampleGeneratorFaceXSeg(SampleGeneratorBase):
    def __init__ (self, paths, pak_names, debug=False, batch_size=1, resolution=256, face_type=None,
                        generators_count=4, ignore_same_path=False, data_format="NHWC",
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False

        samples = sum([ SampleLoader.load (SampleType.FACE, path, pak_name=pak_names[i], ignore_same_path=ignore_same_path) for i, path in enumerate(paths) ]  )
        seg_sample_idxs = SegmentedSampleFilterSubprocessor(samples).run()

        if len(seg_sample_idxs) == 0:
            seg_sample_idxs = SegmentedSampleFilterSubprocessor(samples, count_xseg_mask=True).run()
            if len(seg_sample_idxs) == 0:
                raise Exception(f"未发现 已写遮罩 的图片.")
            else:
                io.log_info(f"使用 {len(seg_sample_idxs)} 张 已写遮罩 的图片.")
        else:
            io.log_info(f"使用 {len(seg_sample_idxs)} 张 手动绘制.")

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        args = (samples, seg_sample_idxs, resolution, face_type, data_format)
        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, args )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, args, start_now=False ) for i in range(self.generators_count) ]

            SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1

        self.initialized = True

    #overridable
    def is_initialized(self):
        return self.initialized

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        samples, seg_sample_idxs, resolution, face_type, data_format = param

        shuffle_idxs = []
        bg_shuffle_idxs = []

        random_flip = True
        rotation_range=[-10,10]
        scale_range=[-0.1, 0.1]
        tx_range=[-0.05, 0.05]
        ty_range=[-0.05, 0.05]

        random_bilinear_resize_chance, random_bilinear_resize_max_size_per = 25,75
        sharpen_chance, sharpen_kernel_max_size = 25, 5
        motion_blur_chance, motion_blur_mb_max_size = 25, 5
        gaussian_blur_chance, gaussian_blur_kernel_max_size = 25, 5
        random_jpeg_compress_chance = 25

        def gen_img_mask(sample):
            img = sample.load_bgr()
            h,w,c = img.shape

            if sample.seg_ie_polys.has_polys():
                mask = np.zeros ((h,w,1), dtype=np.float32)
                sample.seg_ie_polys.overlay_mask(mask)
            elif sample.has_xseg_mask():
                mask = sample.get_xseg_mask()
                mask[mask < 0.5] = 0.0
                mask[mask >= 0.5] = 1.0
            else:
                raise Exception(f'no mask in sample {sample.filename}')

            if face_type == sample.face_type and sample.face_type != FaceType.CUSTOM: # custom always valid for stuff like for wf custom equivivelnet
                if w != resolution:
                    img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 )
                    mask = cv2.resize( mask, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 )
            else:
                mat = LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, face_type)
                img  = cv2.warpAffine( img,  mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )
                mask = cv2.warpAffine( mask, mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )

            if len(mask.shape) == 2:
                mask = mask[...,None]
            return img, mask

        bs = self.batch_size
        while True:
            batches = [ [], [] ]
            filenames = []

            n_batch = 0
            while n_batch < bs:
                try:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = seg_sample_idxs.copy()
                        np.random.shuffle(shuffle_idxs)
                    sample = samples[shuffle_idxs.pop()]
                    img, mask = gen_img_mask(sample)
                    filenames.append(sample.filename)

                    if np.random.randint(2) == 0:
                        if len(bg_shuffle_idxs) == 0:
                            bg_shuffle_idxs = seg_sample_idxs.copy()
                            np.random.shuffle(bg_shuffle_idxs)
                        bg_sample = samples[bg_shuffle_idxs.pop()]

                        bg_img, bg_mask = gen_img_mask(bg_sample)

                        bg_wp   = imagelib.gen_warp_params(resolution, True, rotation_range=[-180,180], scale_range=[-0.10, 0.10], tx_range=[-0.10, 0.10], ty_range=[-0.10, 0.10] )
                        bg_img  = imagelib.warp_by_params (bg_wp, bg_img,  can_warp=False, can_transform=True, can_flip=True, border_replicate=True)
                        bg_mask = imagelib.warp_by_params (bg_wp, bg_mask, can_warp=False, can_transform=True, can_flip=True, border_replicate=False)
                        bg_img = bg_img*(1-bg_mask)
                        if np.random.randint(2) == 0:
                            bg_img = imagelib.apply_random_hsv_shift(bg_img)
                        else:
                            bg_img = imagelib.apply_random_rgb_levels(bg_img)

                        c_mask = 1.0 - (1-bg_mask) * (1-mask)
                        rnd = 0.15 + np.random.uniform()*0.85
                        img = img*(c_mask) + img*(1-c_mask)*rnd + bg_img*(1-c_mask)*(1-rnd)

                    warp_params = imagelib.gen_warp_params(resolution, random_flip, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range )
                    img   = imagelib.warp_by_params (warp_params, img,  can_warp=True, can_transform=True, can_flip=True, border_replicate=True)
                    mask  = imagelib.warp_by_params (warp_params, mask, can_warp=True, can_transform=True, can_flip=True, border_replicate=False)

                    img = np.clip(img.astype(np.float32), 0, 1)
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
                    mask = np.clip(mask, 0, 1)
                    
                    if np.random.randint(2) == 0:
                        # random face flare
                        krn = np.random.randint( resolution//4, resolution )
                        krn = krn - krn % 2 + 1
                        img = img + cv2.GaussianBlur(img*mask, (krn,krn), 0)

                    if np.random.randint(2) == 0:
                        # random bg flare
                        krn = np.random.randint( resolution//4, resolution )
                        krn = krn - krn % 2 + 1
                        img = img + cv2.GaussianBlur(img*(1-mask), (krn,krn), 0)

                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_hsv_shift(img, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:
                        img = imagelib.apply_random_rgb_levels(img, mask=sd.random_circle_faded ([resolution,resolution]))
                        
                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_sharpen( img, sharpen_chance, sharpen_kernel_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:
                        img = imagelib.apply_random_motion_blur( img, motion_blur_chance, motion_blur_mb_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                        img = imagelib.apply_random_gaussian_blur( img, gaussian_blur_chance, gaussian_blur_kernel_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                        
                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_nearest_resize( img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:
                        img = imagelib.apply_random_bilinear_resize( img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = np.clip(img, 0, 1)

                    img = imagelib.apply_random_jpeg_compress( img, random_jpeg_compress_chance, mask=sd.random_circle_faded ([resolution,resolution]))

                    if data_format == "NCHW":
                        img = np.transpose(img, (2,0,1) )
                        mask = np.transpose(mask, (2,0,1) )

                    batches[0].append ( img )
                    batches[1].append ( mask )

                    n_batch += 1
                except:
                    io.log_err ( traceback.format_exc() )

            yield ([ np.array(batch) for batch in batches], filenames)

class SegmentedSampleFilterSubprocessor(Subprocessor):
    #override
    def __init__(self, samples, count_xseg_mask=False ):
        self.samples = samples
        self.samples_len = len(self.samples)
        self.count_xseg_mask = count_xseg_mask

        self.idxs = [*range(self.samples_len)]
        self.result = []
        super().__init__('SegmentedSampleFilterSubprocessor', SegmentedSampleFilterSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(),8)):
            yield 'CPU%d' % (i), {}, {'samples':self.samples, 'count_xseg_mask':self.count_xseg_mask}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Filtering", self.samples_len)

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.idxs) > 0:
            return self.idxs.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.idxs.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        idx, is_ok = result
        if is_ok:
            self.result.append(idx)
        io.progress_bar_inc(1)
    def get_result(self):
        return self.result

    class Cli(Subprocessor.Cli):
        #overridable optional
        def on_initialize(self, client_dict):
            self.samples = client_dict['samples']
            self.count_xseg_mask = client_dict['count_xseg_mask']

        def process_data(self, idx):
            if self.count_xseg_mask:
                return idx, self.samples[idx].has_xseg_mask()
            else:
                return idx, self.samples[idx].seg_ie_polys.get_pts_count() != 0

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
