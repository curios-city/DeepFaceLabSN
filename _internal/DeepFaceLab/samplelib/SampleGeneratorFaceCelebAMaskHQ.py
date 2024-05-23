import multiprocessing
import pickle
import time
import traceback
from enum import IntEnum

import cv2
import numpy as np

from core import imagelib, mplib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import SampleGeneratorBase


class MaskType(IntEnum):
    none   = 0,
    cloth  = 1,
    ear_r  = 2,
    eye_g  = 3,
    hair   = 4,
    hat    = 5,
    l_brow = 6,
    l_ear  = 7,
    l_eye  = 8,
    l_lip  = 9,
    mouth  = 10,
    neck   = 11,
    neck_l = 12,
    nose   = 13,
    r_brow = 14,
    r_ear  = 15,
    r_eye  = 16,
    skin   = 17,
    u_lip  = 18



MaskType_to_name = {
    int(MaskType.none  ) : 'none',
    int(MaskType.cloth ) : 'cloth',
    int(MaskType.ear_r ) : 'ear_r',
    int(MaskType.eye_g ) : 'eye_g',
    int(MaskType.hair  ) : 'hair',
    int(MaskType.hat   ) : 'hat',
    int(MaskType.l_brow) : 'l_brow',
    int(MaskType.l_ear ) : 'l_ear',
    int(MaskType.l_eye ) : 'l_eye',
    int(MaskType.l_lip ) : 'l_lip',
    int(MaskType.mouth ) : 'mouth',
    int(MaskType.neck  ) : 'neck',
    int(MaskType.neck_l) : 'neck_l',
    int(MaskType.nose  ) : 'nose',
    int(MaskType.r_brow) : 'r_brow',
    int(MaskType.r_ear ) : 'r_ear',
    int(MaskType.r_eye ) : 'r_eye',
    int(MaskType.skin  ) : 'skin',
    int(MaskType.u_lip ) : 'u_lip',
}

MaskType_from_name = { MaskType_to_name[k] : k for k in MaskType_to_name.keys() }

class SampleGeneratorFaceCelebAMaskHQ(SampleGeneratorBase):
    def __init__ (self, root_path, debug=False, batch_size=1, resolution=256,
                        generators_count=4, data_format="NHWC",
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False

        dataset_path = root_path / 'CelebAMask-HQ'
        if not dataset_path.exists():
            raise ValueError(f'Unable to find {dataset_path}')

        images_path = dataset_path /'CelebA-HQ-img'
        if not images_path.exists():
            raise ValueError(f'Unable to find {images_path}')

        masks_path = dataset_path / 'CelebAMask-HQ-mask-anno'
        if not masks_path.exists():
            raise ValueError(f'Unable to find {masks_path}')


        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        source_images_paths = pathex.get_image_paths(images_path, return_Path_class=True)
        source_images_paths_len = len(source_images_paths)
        mask_images_paths = pathex.get_image_paths(masks_path, subdirs=True, return_Path_class=True)

        if source_images_paths_len == 0 or len(mask_images_paths) == 0:
            raise ValueError('No training data provided.')

        mask_file_id_hash = {}

        for filepath in io.progress_bar_generator(mask_images_paths, "Loading"):
            stem = filepath.stem

            file_id, mask_type = stem.split('_', 1)
            file_id = int(file_id)

            if file_id not in mask_file_id_hash:
                mask_file_id_hash[file_id] = {}

            mask_file_id_hash[file_id][ MaskType_from_name[mask_type] ] = str(filepath.relative_to(masks_path))

        source_file_id_set = set()

        for filepath in source_images_paths:
            stem = filepath.stem

            file_id = int(stem)
            source_file_id_set.update ( {file_id} )

        for k in mask_file_id_hash.keys():
            if k not in source_file_id_set:
                io.log_err (f"Corrupted dataset: {k} not in {images_path}")



        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (images_path, masks_path, mask_file_id_hash, data_format) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (images_path, masks_path, mask_file_id_hash, data_format), start_now=False ) \
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
        images_path, masks_path, mask_file_id_hash, data_format = param

        file_ids = list(mask_file_id_hash.keys())

        shuffle_file_ids = []

        resolution = 256
        random_flip = True
        rotation_range=[-15,15]
        scale_range=[-0.10, 0.95]
        tx_range=[-0.3, 0.3]
        ty_range=[-0.3, 0.3]

        random_bilinear_resize = (25,75)
        motion_blur = (25, 5)
        gaussian_blur = (25, 5)

        bs = self.batch_size
        while True:
            batches = None

            n_batch = 0
            while n_batch < bs:
                try:
                    if len(shuffle_file_ids) == 0:
                        shuffle_file_ids = file_ids.copy()
                        np.random.shuffle(shuffle_file_ids)

                    file_id = shuffle_file_ids.pop()
                    masks = mask_file_id_hash[file_id]
                    image_path = images_path / f'{file_id}.jpg'

                    skin_path = masks.get(MaskType.skin, None)
                    hair_path = masks.get(MaskType.hair, None)
                    hat_path = masks.get(MaskType.hat, None)
                    #neck_path = masks.get(MaskType.neck, None)

                    img = cv2_imread(image_path).astype(np.float32) / 255.0
                    mask = cv2_imread(masks_path / skin_path)[...,0:1].astype(np.float32) / 255.0

                    if hair_path is not None:
                        hair_path = masks_path / hair_path
                        if hair_path.exists():
                            hair = cv2_imread(hair_path)[...,0:1].astype(np.float32) / 255.0
                            mask *= (1-hair)

                    if hat_path is not None:
                        hat_path = masks_path / hat_path
                        if hat_path.exists():
                            hat = cv2_imread(hat_path)[...,0:1].astype(np.float32) / 255.0
                            mask *= (1-hat)
                    
                    #if neck_path is not None:
                    #    neck_path = masks_path / neck_path
                    #    if neck_path.exists():
                    #        neck = cv2_imread(neck_path)[...,0:1].astype(np.float32) / 255.0
                    #        mask = np.clip(mask+neck, 0, 1)
                            
                    warp_params = imagelib.gen_warp_params(resolution, random_flip, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range )
  
                    img = cv2.resize( img, (resolution,resolution), cv2.INTER_LANCZOS4 )
                    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                    h = ( h + np.random.randint(360) ) % 360
                    s = np.clip ( s + np.random.random()-0.5, 0, 1 )
                    v = np.clip ( v + np.random.random()/2-0.25, 0, 1 )                    
                    img = np.clip( cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR) , 0, 1 )
                            
                    if motion_blur is not None:
                        chance, mb_max_size = motion_blur
                        chance = np.clip(chance, 0, 100)

                        mblur_rnd_chance = np.random.randint(100)
                        mblur_rnd_kernel = np.random.randint(mb_max_size)+1
                        mblur_rnd_deg    = np.random.randint(360)

                        if mblur_rnd_chance < chance:
                            img = imagelib.LinearMotionBlur (img, mblur_rnd_kernel, mblur_rnd_deg )

                    img = imagelib.warp_by_params (warp_params, img,  can_warp=True, can_transform=True, can_flip=True, border_replicate=False, cv2_inter=cv2.INTER_LANCZOS4)
                    
                    if gaussian_blur is not None:
                        chance, kernel_max_size = gaussian_blur
                        chance = np.clip(chance, 0, 100)

                        gblur_rnd_chance = np.random.randint(100)
                        gblur_rnd_kernel = np.random.randint(kernel_max_size)*2+1

                        if gblur_rnd_chance < chance:
                            img = cv2.GaussianBlur(img, (gblur_rnd_kernel,) *2 , 0)
                            
                    if random_bilinear_resize is not None:
                        chance, max_size_per = random_bilinear_resize
                        chance = np.clip(chance, 0, 100)                        
                        pick_chance = np.random.randint(100)                        
                        resize_to = resolution - int( np.random.rand()* int(resolution*(max_size_per/100.0)) )                        
                        img = cv2.resize (img, (resize_to,resize_to), cv2.INTER_LINEAR )
                        img = cv2.resize (img, (resolution,resolution), cv2.INTER_LINEAR )
                        
                            
                    mask = cv2.resize( mask, (resolution,resolution), cv2.INTER_LANCZOS4 )[...,None]
                    mask = imagelib.warp_by_params (warp_params, mask, can_warp=True, can_transform=True, can_flip=True, border_replicate=False, cv2_inter=cv2.INTER_LANCZOS4)
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
                    mask = np.clip(mask, 0, 1)

                    if data_format == "NCHW":
                        img = np.transpose(img, (2,0,1) )
                        mask = np.transpose(mask, (2,0,1) )
                        
                    if batches is None:
                        batches = [ [], [] ]
                    
                    batches[0].append ( img )
                    batches[1].append ( mask )

                    n_batch += 1
                except:
                    io.log_err ( traceback.format_exc() )

            yield [ np.array(batch) for batch in batches]
