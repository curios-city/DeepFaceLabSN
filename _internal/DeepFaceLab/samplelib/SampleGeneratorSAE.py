import multiprocessing
import time
import traceback

import cv2
import numpy as np
import numpy.linalg as npla

from core import mplib
from core import imagelib
from core.interact import interact as io
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from core import mathlib
from facelib import LandmarksProcessor, FaceType
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)

class SampleGeneratorSAE(SampleGeneratorBase):
    def __init__ (self, src_samples_path, dst_samples_path,
                        resolution,
                        face_type,
                        random_src_flip=False,
                        random_dst_flip=False,
                        ct_mode=None,
                        uniform_yaw_distribution=False,
                        data_format='NHWC',
                        debug=False, batch_size=1,
                        raise_on_no_data=True,
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False
        self.resolution = resolution
        self.face_type = face_type
        self.random_src_flip = random_src_flip
        self.random_dst_flip = random_dst_flip
        self.ct_mode = ct_mode
        self.data_format = data_format

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = 8

        src_samples = SampleLoader.load (SampleType.FACE, src_samples_path)
        src_samples_len = len(src_samples)

        if src_samples_len == 0:
            raise ValueError(f'No samples in {src_samples_path}')

        dst_samples = SampleLoader.load (SampleType.FACE, dst_samples_path)
        dst_samples_len = len(dst_samples)

        if dst_samples_len == 0:
            raise ValueError(f'No samples in {dst_samples_path}')

        if uniform_yaw_distribution:
            src_index_host = self._filter_uniform_yaw(src_samples)
            dst_index_host = self._filter_uniform_yaw(dst_samples)
        else:
            src_index_host = mplib.IndexHost(src_samples_len)
            dst_index_host = mplib.IndexHost(dst_samples_len)

        ct_index_host = mplib.IndexHost(dst_samples_len) if ct_mode is not None else None
    
        self.comm_qs = [  multiprocessing.Queue() for i in range(self.generators_count) ]
            
        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (self.comm_qs[0], src_samples, dst_samples, src_index_host.create_cli(), dst_index_host.create_cli(), ct_index_host.create_cli() if ct_index_host is not None else None) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (self.comm_qs[i], src_samples, dst_samples, src_index_host.create_cli(), dst_index_host.create_cli(), ct_index_host.create_cli() if ct_index_host is not None else None), start_now=False ) \
                               for i in range(self.generators_count) ]

        self.generator_counter = -1

        self.initialized = True
        
    def start(self):
        if not self.debug:
            SubprocessGenerator.start_in_parallel( self.generators )

    def _filter_uniform_yaw(self, samples):
        samples_pyr = [ ( idx, sample.get_pitch_yaw_roll() ) for idx, sample in enumerate(samples) ]

        grads = 128
        #instead of math.pi / 2, using -1.2,+1.2 because actually maximum yaw for 2DFAN landmarks are -1.2+1.2
        grads_space = np.linspace (-1.2, 1.2,grads)

        yaws_sample_list = [None]*grads
        for g in io.progress_bar_generator ( range(grads), "Sort by yaw"):
            yaw = grads_space[g]
            next_yaw = grads_space[g+1] if g < grads-1 else yaw

            yaw_samples = []
            for idx, pyr in samples_pyr:
                s_yaw = -pyr[1]
                if (g == 0          and s_yaw < next_yaw) or \
                (g < grads-1     and s_yaw >= yaw and s_yaw < next_yaw) or \
                (g == grads-1    and s_yaw >= yaw):
                    yaw_samples += [ idx ]
            if len(yaw_samples) > 0:
                yaws_sample_list[g] = yaw_samples

        yaws_sample_list = [ y for y in yaws_sample_list if y is not None ]

        return mplib.Index2DHost( yaws_sample_list )
        
    def set_face_scale(self, scale):
        for comm_q in self.comm_qs:
            comm_q.put( ('face_scale', scale) )
     
     
    #overridable
    def is_initialized(self):
        return self.initialized

    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            return []

        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        comm_q, src_samples, dst_samples, src_index_host, dst_index_host, ct_index_host = param

        batch_size = self.batch_size
        resolution = self.resolution
        face_type = self.face_type
        data_format = self.data_format
        random_src_flip = self.random_src_flip
        random_dst_flip = self.random_dst_flip
        ct_mode = self.ct_mode

        rotation_range=[-10,10]
        scale_range=[-0.05, 0.05]
        tx_range=[-0.05, 0.05]
        ty_range=[-0.05, 0.05]
        rnd_state = np.random

        face_scale = 1.0
        
        hi_res = 1024

        def gen_sample(sample, target_face_type, resolution, allow_flip=False, scale=1.0, ct_mode=None, ct_sample=None):#:, tx, ty, rotation, scale):
            tx = rnd_state.uniform( tx_range[0], tx_range[1] )
            ty = rnd_state.uniform( ty_range[0], ty_range[1] )
            rotation = rnd_state.uniform( rotation_range[0], rotation_range[1] )
            scale = rnd_state.uniform(scale +scale_range[0], scale +scale_range[1])
            
            flip = allow_flip and rnd_state.randint(10) < 4

            face_type = sample.face_type
            face_lmrks = sample.landmarks
            face = sample.load_bgr()
            h,w,c = face.shape

            if face_type == FaceType.HEAD:
                hi_mat = LandmarksProcessor.get_transform_mat (face_lmrks, hi_res, FaceType.HEAD)
            else:
                hi_mat = LandmarksProcessor.get_transform_mat (face_lmrks, hi_res, FaceType.HEAD_FACE)

            hi_lmrks = LandmarksProcessor.transform_points(face_lmrks, hi_mat)
            hi_warp_params = imagelib.gen_warp_params(hi_res)
            face_warp_params = imagelib.gen_warp_params(resolution)

            hi_to_target_mat = LandmarksProcessor.get_transform_mat (hi_lmrks, resolution, target_face_type)
            hi_to_target_mat = mathlib.transform_mat(hi_to_target_mat, resolution, tx, ty, rotation, scale)

            face_to_target_mat = LandmarksProcessor.get_transform_mat (face_lmrks, resolution, target_face_type)
            face_to_target_mat = mathlib.transform_mat(face_to_target_mat, resolution, tx, ty, rotation, scale)

            warped_face = face
            if ct_mode is not None:
                ct_bgr = ct_sample.load_bgr()
                ct_bgr = cv2.resize(ct_bgr, (w,h), interpolation=cv2.INTER_LINEAR )
                warped_face = imagelib.color_transfer (ct_mode, warped_face, ct_bgr)

            warped_face = cv2.warpAffine(warped_face, hi_mat, (hi_res,hi_res), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )
            warped_face = np.clip( imagelib.warp_by_params (hi_warp_params, warped_face, can_warp=True, can_transform=False, can_flip=False, border_replicate=cv2.BORDER_REPLICATE), 0, 1)
            warped_face = cv2.warpAffine(warped_face, hi_to_target_mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )

            """
            if face_type != target_face_type:
                ...
            else:
                if w != resolution:
                    face = cv2.resize(face, (resolution, resolution), interpolation=cv2.INTER_CUBIC )
            """                 
                
            # warped_face = cv2.warpAffine(warped_face, face_to_target_mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )
            # warped_face = np.clip( imagelib.warp_by_params (face_warp_params, warped_face, can_warp=True, can_transform=False, can_flip=False, border_replicate=cv2.BORDER_REPLICATE), 0, 1)

            target_face = face
            if ct_mode is not None:
                target_face = imagelib.color_transfer (ct_mode, target_face, ct_bgr)

            target_face = cv2.warpAffine(target_face, face_to_target_mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )


            face_mask = sample.get_xseg_mask()
            if face_mask is not None:
                if face_mask.shape[0] != h or face_mask.shape[1] != w:
                    face_mask = cv2.resize(face_mask, (w,h), interpolation=cv2.INTER_CUBIC)
                    face_mask = imagelib.normalize_channels(face_mask, 1)
            else:
                face_mask = LandmarksProcessor.get_image_hull_mask (face.shape, face_lmrks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
            face_mask = np.clip(face_mask, 0, 1)

            target_face_mask = cv2.warpAffine(face_mask, face_to_target_mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR )
            target_face_mask = imagelib.normalize_channels(target_face_mask, 1)
            target_face_mask = np.clip(target_face_mask, 0, 1)
                
            em_mask = np.clip(LandmarksProcessor.get_image_eye_mask (face.shape, face_lmrks) + \
                              LandmarksProcessor.get_image_mouth_mask (face.shape, face_lmrks), 0, 1)

            target_face_em = cv2.warpAffine(em_mask, face_to_target_mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR )
            target_face_em = imagelib.normalize_channels(target_face_em, 1)
            
            div = target_face_em.max()
            if div != 0.0:
                target_face_em = target_face_em / div
            
            target_face_em = target_face_em * target_face_mask

            # while True:
            #     cv2.imshow('', warped_face)
            #     cv2.waitKey(0)

            #     cv2.imshow('', target_face)
            #     cv2.waitKey(0)

            #     cv2.imshow('', target_face_mask)
            #     cv2.waitKey(0)

            #     cv2.imshow('', target_face_em)
            #     cv2.waitKey(0)
            # import code
            # code.interact(local=dict(globals(), **locals()))
            
            if flip:
                warped_face = warped_face[:,::-1,...]
                target_face = target_face[:,::-1,...]
                target_face_mask = target_face_mask[:,::-1,...]
                target_face_em = target_face_em[:,::-1,...]

            return warped_face, target_face, target_face_mask, target_face_em

        
        while True:
            while not comm_q.empty():
                cmd, param = comm_q.get()
                if cmd == 'face_scale':
                    face_scale = param   
                    
            batches = [ [], [], [], [], [], [] ,[] ,[] ] #

            src_indexes = src_index_host.multi_get(batch_size)
            dst_indexes = dst_index_host.multi_get(batch_size)

            for n_batch in range(batch_size):
                src_sample = src_samples[src_indexes[n_batch]]
                dst_sample = dst_samples[dst_indexes[n_batch]]

                src_warped_face, src_target_face, src_target_face_mask, src_target_face_em = \
                    gen_sample(src_sample, face_type, resolution, allow_flip=random_src_flip, scale=face_scale, ct_mode=ct_mode, ct_sample=dst_sample)

                dst_warped_face, dst_target_face, dst_target_face_mask, dst_target_face_em = \
                    gen_sample(dst_sample, face_type, resolution, allow_flip=random_dst_flip, scale=face_scale)



                if data_format == "NCHW":
                    src_warped_face = np.transpose(src_warped_face, (2,0,1) )
                    src_target_face = np.transpose(src_target_face, (2,0,1) )
                    src_target_face_mask = np.transpose(src_target_face_mask, (2,0,1) )
                    src_target_face_em = np.transpose(src_target_face_em, (2,0,1) )
                    dst_warped_face = np.transpose(dst_warped_face, (2,0,1) )
                    dst_target_face = np.transpose(dst_target_face, (2,0,1) )
                    dst_target_face_mask = np.transpose(dst_target_face_mask, (2,0,1) )
                    dst_target_face_em = np.transpose(dst_target_face_em, (2,0,1) )

                batches[0].append(src_warped_face)
                batches[1].append(src_target_face)
                batches[2].append(src_target_face_mask)
                batches[3].append(src_target_face_em)
                batches[4].append(dst_warped_face)
                batches[5].append(dst_target_face)
                batches[6].append(dst_target_face_mask)
                batches[7].append(dst_target_face_em)


            yield [ np.array(batch) for batch in batches]
