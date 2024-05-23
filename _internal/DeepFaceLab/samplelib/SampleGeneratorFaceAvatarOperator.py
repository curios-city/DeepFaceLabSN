import multiprocessing
import pickle
import time
import traceback
from enum import IntEnum

import cv2
import numpy as np

from core import imagelib, mplib, pathex
from core.imagelib import sd
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor, SampleType)

class SampleGeneratorFaceAvatarOperator(SampleGeneratorBase):
    def __init__ (self, root_path, debug=False, batch_size=1, resolution=256, face_type=None,
                        generators_count=4, data_format="NHWC",
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False


        dataset_path = root_path / 'AvatarOperatorDataset'
        if not dataset_path.exists():
            raise ValueError(f'Unable to find {dataset_path}')

        chains_dir_names = pathex.get_all_dir_names(dataset_path)

        samples = SampleLoader.load (SampleType.FACE, dataset_path, subdirs=True)
        sample_idx_by_path = { sample.filename : i for i,sample in enumerate(samples) }

        kf_idxs = []

        for chain_dir_name in chains_dir_names:
            chain_root_path = dataset_path / chain_dir_name

            subchain_dir_names = pathex.get_all_dir_names(chain_root_path)
            try:
                subchain_dir_names.sort(key=int)
            except:
                raise Exception(f'{chain_root_path} must contain only numerical name of directories')
            chain_samples = []

            for subchain_dir_name in subchain_dir_names:
                subchain_root = chain_root_path / subchain_dir_name
                subchain_samples = [  sample_idx_by_path[image_path] for image_path in pathex.get_image_paths(subchain_root) \
                                                                     if image_path in sample_idx_by_path ]

                if len(subchain_samples) < 3:
                    raise Exception(f'subchain {subchain_dir_name} must contain at least 3 faces. If you delete this subchain, then th echain will be corrupted.')

                chain_samples += [ subchain_samples ]

            chain_samples_len = len(chain_samples)
            for i in range(chain_samples_len-1):
                kf_idxs += [ ( chain_samples[i+1][0], chain_samples[i][-1], chain_samples[i][:-1] ) ]
                
            for i in range(1,chain_samples_len):
                kf_idxs += [ ( chain_samples[i-1][-1], chain_samples[i][0], chain_samples[i][1:]  ) ]

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (samples, kf_idxs, resolution, face_type, data_format) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (samples, kf_idxs, resolution, face_type, data_format), start_now=False ) \
                               for i in range(self.generators_count) ]

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
        samples, kf_idxs, resolution, face_type, data_format = param
        
        kf_idxs_len = len(kf_idxs)

        shuffle_idxs = []
        idxs = [*range(len(samples))]

        random_flip = True
        rotation_range=[-10,10]
        scale_range=[-0.05, 0.05]
        tx_range=[-0.05, 0.05]
        ty_range=[-0.05, 0.05]

        bs = self.batch_size
        while True:
            batches = [ [], [] , [], [], [], [] ]

            n_batch = 0
            while n_batch < bs:
                try:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = idxs.copy()
                        np.random.shuffle(shuffle_idxs)
                    idx = shuffle_idxs.pop()


                    key_idx, key_chain_idx, chain_idxs = kf_idxs[ np.random.randint(kf_idxs_len) ]
                    
                    key_sample = samples[key_idx]
                    key_chain_sample = samples[key_chain_idx]
                    chain_sample = samples[ chain_idxs[np.random.randint(len(chain_idxs)) ] ]
                    
                    #print('==========')
                    #print(key_sample.filename)
                    #print(key_chain_sample.filename)
                    #print(chain_sample.filename)
                    
                    sample = samples[idx]

                    img = sample.load_bgr()
                    
                    key_img = key_sample.load_bgr()
                    key_chain_img = key_chain_sample.load_bgr()
                    chain_img = chain_sample.load_bgr()
                    
                    h,w,c = img.shape

                    mask = LandmarksProcessor.get_image_hull_mask (img.shape, sample.landmarks)
                    mask = np.clip(mask, 0, 1)
                
                    warp_params = imagelib.gen_warp_params(resolution, random_flip, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range )

                    if face_type == sample.face_type:
                        if w != resolution:
                            img = cv2.resize( img, (resolution, resolution), cv2.INTER_CUBIC )
                            key_img = cv2.resize( key_img, (resolution, resolution), cv2.INTER_CUBIC )
                            key_chain_img = cv2.resize( key_chain_img, (resolution, resolution), cv2.INTER_CUBIC )
                            chain_img = cv2.resize( chain_img, (resolution, resolution), cv2.INTER_CUBIC )
                            
                            mask = cv2.resize( mask, (resolution, resolution), cv2.INTER_CUBIC )
                    else:
                        mat = LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, face_type)
                        img  = cv2.warpAffine( img,  mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )
                        key_img  = cv2.warpAffine( key_img,  mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )
                        key_chain_img  = cv2.warpAffine( key_chain_img,  mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )
                        chain_img  = cv2.warpAffine( chain_img,  mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )
                        mask = cv2.warpAffine( mask, mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_CUBIC )

                    if len(mask.shape) == 2:
                        mask = mask[...,None]

                    img_warped      = imagelib.warp_by_params (warp_params, img,  can_warp=True, can_transform=True, can_flip=True, border_replicate=True)                    
                    img_transformed = imagelib.warp_by_params (warp_params, img,  can_warp=False, can_transform=True, can_flip=True, border_replicate=True)
                    
                    mask  = imagelib.warp_by_params (warp_params, mask, can_warp=True, can_transform=True, can_flip=True, border_replicate=False)

                    key_img        = imagelib.warp_by_params (warp_params, key_img,  can_warp=False, can_transform=False, can_flip=False, border_replicate=True)
                    key_chain_img  = imagelib.warp_by_params (warp_params, key_chain_img,  can_warp=False, can_transform=False, can_flip=False, border_replicate=True)
                    chain_img      = imagelib.warp_by_params (warp_params, chain_img,  can_warp=False, can_transform=False, can_flip=False, border_replicate=True)
                    
                    
                    img_warped = np.clip(img_warped.astype(np.float32), 0, 1)
                    img_transformed = np.clip(img_transformed.astype(np.float32), 0, 1)
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
                    mask = np.clip(mask, 0, 1)

                    if data_format == "NCHW":
                        img_warped = np.transpose(img_warped, (2,0,1) )
                        img_transformed = np.transpose(img_transformed, (2,0,1) )
                        mask = np.transpose(mask, (2,0,1) )
                        
                        key_img = np.transpose(key_img, (2,0,1) )
                        key_chain_img = np.transpose(key_chain_img, (2,0,1) )
                        chain_img = np.transpose(chain_img, (2,0,1) )

                    batches[0].append ( img_warped )
                    batches[1].append ( img_transformed )
                    batches[2].append ( mask )
                    batches[3].append ( key_img )
                    batches[4].append ( key_chain_img )
                    batches[5].append ( chain_img )

                    n_batch += 1
                except:
                    io.log_err ( traceback.format_exc() )

            yield [ np.array(batch) for batch in batches]
