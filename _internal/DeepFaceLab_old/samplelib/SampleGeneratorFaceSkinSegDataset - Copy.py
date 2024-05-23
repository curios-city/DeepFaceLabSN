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

class SampleGeneratorFaceSkinSegDataset(SampleGeneratorBase):
    def __init__ (self, root_path, debug=False, batch_size=1, resolution=256, face_type=None,
                        generators_count=4, data_format="NHWC",
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False


        aligned_path = root_path /'aligned'
        if not aligned_path.exists():
            raise ValueError(f'Unable to find {aligned_path}')

        obstructions_path = root_path / 'obstructions'

        obstructions_images_paths = pathex.get_image_paths(obstructions_path, image_extensions=['.png'], subdirs=True)

        samples = SampleLoader.load (SampleType.FACE, aligned_path, subdirs=True)
        self.samples_len = len(samples)

        pickled_samples = pickle.dumps(samples, 4)

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (pickled_samples, obstructions_images_paths, resolution, face_type, data_format) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (pickled_samples, obstructions_images_paths, resolution, face_type, data_format), start_now=False ) \
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
        pickled_samples, obstructions_images_paths, resolution, face_type, data_format = param

        samples = pickle.loads(pickled_samples)

        obstructions_images_paths_len = len(obstructions_images_paths)
        shuffle_o_idxs = []
        o_idxs = [*range(obstructions_images_paths_len)]

        shuffle_idxs = []
        idxs = [*range(len(samples))]

        random_flip = True
        rotation_range=[-10,10]
        scale_range=[-0.05, 0.05]
        tx_range=[-0.05, 0.05]
        ty_range=[-0.05, 0.05]

        o_random_flip = True
        o_rotation_range=[-180,180]
        o_scale_range=[-0.5, 0.05]
        o_tx_range=[-0.5, 0.5]
        o_ty_range=[-0.5, 0.5]

        random_bilinear_resize_chance, random_bilinear_resize_max_size_per = 25,75
        motion_blur_chance, motion_blur_mb_max_size = 25, 5
        gaussian_blur_chance, gaussian_blur_kernel_max_size = 25, 5

        bs = self.batch_size
        while True:
            batches = [ [], [] ]

            n_batch = 0
            while n_batch < bs:
                try:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = idxs.copy()
                        np.random.shuffle(shuffle_idxs)

                    idx = shuffle_idxs.pop()

                    sample = samples[idx]

                    img = sample.load_bgr()
                    h,w,c = img.shape

                    mask = np.zeros ((h,w,1), dtype=np.float32)
                    sample.ie_polys.overlay_mask(mask)

                    warp_params = imagelib.gen_warp_params(resolution, random_flip, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range )

                    if face_type == sample.face_type:
                        if w != resolution:
                            img = cv2.resize( img, (resolution, resolution), cv2.INTER_LANCZOS4 )
                            mask = cv2.resize( mask, (resolution, resolution), cv2.INTER_LANCZOS4 )
                    else:
                        mat = LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, face_type)
                        img  = cv2.warpAffine( img,  mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )
                        mask = cv2.warpAffine( mask, mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )

                    if len(mask.shape) == 2:
                        mask = mask[...,None]

                    if obstructions_images_paths_len != 0:
                        # apply obstruction
                        if len(shuffle_o_idxs) == 0:
                            shuffle_o_idxs = o_idxs.copy()
                            np.random.shuffle(shuffle_o_idxs)
                        o_idx = shuffle_o_idxs.pop()
                        o_img = cv2_imread (obstructions_images_paths[o_idx]).astype(np.float32) / 255.0
                        oh,ow,oc = o_img.shape
                        if oc == 4:
                            ohw = max(oh,ow)
                            scale = resolution / ohw

                            #o_img = cv2.resize (o_img, ( int(ow*rate), int(oh*rate),  ), cv2.INTER_CUBIC)





                            mat = cv2.getRotationMatrix2D( (ow/2,oh/2),
                                                        np.random.uniform( o_rotation_range[0], o_rotation_range[1] ),
                                                        1.0 )

                            mat += np.float32( [[0,0, -ow/2 ],
                                                [0,0, -oh/2 ]])
                            mat *= scale * np.random.uniform(1 +o_scale_range[0], 1 +o_scale_range[1])
                            mat += np.float32( [[0, 0, resolution/2 + resolution*np.random.uniform( o_tx_range[0], o_tx_range[1] ) ],
                                                [0, 0, resolution/2 + resolution*np.random.uniform( o_ty_range[0], o_ty_range[1] ) ] ])


                            o_img  = cv2.warpAffine( o_img,  mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )

                            if o_random_flip and np.random.randint(10) < 4:
                                o_img = o_img[:,::-1,...]

                            o_mask = o_img[...,3:4]
                            o_mask[o_mask>0] = 1.0


                            o_mask = cv2.erode (o_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
                            o_mask = cv2.GaussianBlur(o_mask, (5, 5) , 0)[...,None]

                            img = img*(1-o_mask) + o_img[...,0:3]*o_mask

                            o_mask[o_mask<0.5] = 0.0


                            #import code
                            #code.interact(local=dict(globals(), **locals()))
                            mask *= (1-o_mask)


                            #cv2.imshow ("", np.clip(o_img*255, 0,255).astype(np.uint8) )
                            #cv2.waitKey(0)


                    img   = imagelib.warp_by_params (warp_params, img,  can_warp=True, can_transform=True, can_flip=True, border_replicate=False)
                    mask  = imagelib.warp_by_params (warp_params, mask, can_warp=True, can_transform=True, can_flip=True, border_replicate=False)


                    img = np.clip(img.astype(np.float32), 0, 1)
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
                    mask = np.clip(mask, 0, 1)


                    img = imagelib.apply_random_hsv_shift(img, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = imagelib.apply_random_motion_blur( img, motion_blur_chance, motion_blur_mb_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = imagelib.apply_random_gaussian_blur( img, gaussian_blur_chance, gaussian_blur_kernel_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = imagelib.apply_random_bilinear_resize( img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded ([resolution,resolution]))

                    if data_format == "NCHW":
                        img = np.transpose(img, (2,0,1) )
                        mask = np.transpose(mask, (2,0,1) )

                    batches[0].append ( img )
                    batches[1].append ( mask )

                    n_batch += 1
                except:
                    io.log_err ( traceback.format_exc() )

            yield [ np.array(batch) for batch in batches]
