import math
from enum import IntEnum
from core.imagelib.shadows import shadow_highlights_augmentation

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.imagelib import LinearMotionBlur
from core.imagelib.color_transfer import random_lab_rotation
from facelib import FaceType, LandmarksProcessor


class SampleProcessor(object):
    class SampleType(IntEnum):
        NONE = 0
        IMAGE = 1
        FACE_IMAGE = 2
        FACE_MASK  = 3
        LANDMARKS_ARRAY            = 4
        PITCH_YAW_ROLL             = 5
        PITCH_YAW_ROLL_SIGMOID     = 6

    class ChannelType(IntEnum):
        NONE = 0
        BGR                   = 1  #BGR
        G                     = 2  #Grayscale
        GGG                   = 3  #3xGrayscale
        LAB_RAND_TRANSFORM    = 4  # LAB random transform


    class FaceMaskType(IntEnum):
        NONE           = 0
        FULL_FACE      = 1  # mask all hull as grayscale
        EYES           = 2  # mask eyes hull as grayscale
        EYES_MOUTH     = 3  # eyes and mouse

    class Options(object):
        def __init__(self, random_flip = True, rotation_range=[-3,3], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05] ):
            self.random_flip = random_flip
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range
            #print("test super warp",self.rotation_range,self.scale_range)

    @staticmethod
    def process (samples, sample_process_options, output_sample_types, debug, ct_sample=None):
        SPST = SampleProcessor.SampleType
        SPCT = SampleProcessor.ChannelType
        SPFMT = SampleProcessor.FaceMaskType

        outputs = []
        for sample in samples:
            sample_rnd_seed = np.random.randint(0x80000000)

            sample_face_type = sample.face_type
            sample_bgr = sample.load_bgr()
            sample_landmarks = sample.landmarks
            ct_sample_bgr = None
            h,w,c = sample_bgr.shape

            def get_full_face_mask():
                xseg_mask = sample.get_xseg_mask()
                if xseg_mask is not None:
                    if xseg_mask.shape[0] != h or xseg_mask.shape[1] != w:
                        xseg_mask = cv2.resize(xseg_mask, (w,h), interpolation=cv2.INTER_CUBIC)
                        xseg_mask = imagelib.normalize_channels(xseg_mask, 1)
                    return np.clip(xseg_mask, 0, 1)
                else:
                    full_face_mask = LandmarksProcessor.get_image_hull_mask (sample_bgr.shape, sample_landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
                    return np.clip(full_face_mask, 0, 1)

            def get_eyes_mask():
                eyes_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample_landmarks)
                # set eye masks to 1-2
                clip = np.clip(eyes_mask, 0, 1)
                clip[clip > 0.1] += 1
                return clip

            def get_mouth_mask():
                mouth_mask = LandmarksProcessor.get_image_mouth_mask (sample_bgr.shape, sample_landmarks)
                # set eye masks to 2-3
                clip = np.clip(mouth_mask, 0, 1)
                clip[clip > 0.1] += 2
                return clip

            is_face_sample = sample_landmarks is not None

            if debug and is_face_sample:
                LandmarksProcessor.draw_landmarks (sample_bgr, sample_landmarks, (0, 1, 0))
            outputs_sample = []
            for opts in output_sample_types:
                resolution     = opts.get('resolution', 0)
                sample_type    = opts.get('sample_type', SPST.NONE)
                channel_type   = opts.get('channel_type', SPCT.NONE)
                nearest_resize_to = opts.get('nearest_resize_to', None)
                warp           = opts.get('warp', False)
                transform      = opts.get('transform', False)
                random_hsv_shift_amount = opts.get('random_hsv_shift_amount', 0)
                random_downsample = opts.get('random_downsample', False)
                random_noise = opts.get('random_noise', False)
                random_blur = opts.get('random_blur', False)
                random_jpeg = opts.get('random_jpeg', False)
                random_shadow = opts.get('random_shadow', False)
                normalize_tanh = opts.get('normalize_tanh', False)
                ct_mode        = opts.get('ct_mode', None)
                data_format    = opts.get('data_format', 'NHWC')
                rnd_seed_shift      = opts.get('rnd_seed_shift', 0)
                warp_rnd_seed_shift = opts.get('warp_rnd_seed_shift', rnd_seed_shift)

                rnd_state      = np.random.RandomState (sample_rnd_seed+rnd_seed_shift)
                warp_rnd_state = np.random.RandomState (sample_rnd_seed+warp_rnd_seed_shift)

                warp_params = imagelib.gen_warp_params(resolution,
                                                       sample_process_options.random_flip,
                                                       rotation_range=sample_process_options.rotation_range,
                                                       scale_range=sample_process_options.scale_range,
                                                       tx_range=sample_process_options.tx_range,
                                                       ty_range=sample_process_options.ty_range,
                                                       rnd_state=rnd_state,
                                                       warp_rnd_state=warp_rnd_state,
                                                       )

                if sample_type == SPST.FACE_MASK or sample_type == SPST.IMAGE:
                    border_replicate = False
                elif sample_type == SPST.FACE_IMAGE:
                    border_replicate = True


                border_replicate = opts.get('border_replicate', border_replicate)
                borderMode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT


                if sample_type == SPST.FACE_IMAGE or sample_type == SPST.FACE_MASK:
                    if not is_face_sample:
                        raise ValueError("face_samples should be provided for sample_type FACE_*")

                if sample_type == SPST.FACE_IMAGE or sample_type == SPST.FACE_MASK:
                    face_type      = opts.get('face_type', None)
                    face_mask_type = opts.get('face_mask_type', SPFMT.NONE)

                    if face_type is None:
                        raise ValueError("face_type must be defined for face samples")

                    if sample_type == SPST.FACE_MASK:
                        if face_mask_type == SPFMT.FULL_FACE:
                            img = get_full_face_mask()
                        elif face_mask_type == SPFMT.EYES:
                            img = get_eyes_mask()
                        elif face_mask_type == SPFMT.EYES_MOUTH:
                            # sets both eyes and mouth mask parts
                            img = get_full_face_mask()
                            mask = img.copy()
                            mask[mask != 0.0] = 1.0
                            eye_mask = get_eyes_mask() * mask
                            img = np.where(eye_mask > 1, eye_mask, img)

                            mouth_mask = get_mouth_mask() * mask
                            img = np.where(mouth_mask > 2, mouth_mask, img)
                        else:
                            img = np.zeros ( sample_bgr.shape[0:2]+(1,), dtype=np.float32)

                        if sample_face_type == FaceType.MARK_ONLY:
                            raise NotImplementedError()
                            mat  = LandmarksProcessor.get_transform_mat (sample_landmarks, warp_resolution, face_type)
                            img = cv2.warpAffine( img, mat, (warp_resolution, warp_resolution), flags=cv2.INTER_LINEAR )
                        else:
                            if face_type != sample_face_type and sample_face_type != FaceType.CUSTOM: # custom always valid for stuff like for wf custom equivalent
                                mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, face_type)
                                img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_LINEAR )
                            else:
                                if w != resolution:
                                    img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_LINEAR )

                            img = imagelib.warp_by_params (warp_params, img, warp, transform, can_flip=True, border_replicate=border_replicate, cv2_inter=cv2.INTER_LINEAR)

                        if len(img.shape) == 2:
                            img = img[...,None]

                        if channel_type == SPCT.G:
                            out_sample = img.astype(np.float32)
                        else:
                            raise ValueError("only channel_type.G supported for the mask")

                    elif sample_type == SPST.FACE_IMAGE:
                        img = sample_bgr

                        if face_type != sample_face_type and sample_face_type != FaceType.CUSTOM:
                            mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, face_type)
                            img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_CUBIC )
                        else:
                            if w != resolution:
                                img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_CUBIC )

                        # Apply random color transfer
                        if ct_mode is not None and (ct_sample is not None or ct_mode == 'fs-aug' or ct_mode == 'cc-aug'):
                            if ct_mode == 'fs-aug':
                                img = imagelib.color_augmentation(img, sample_rnd_seed)
                            elif ct_mode == 'cc-aug':
                                img = imagelib.cc_aug(img, sample_rnd_seed)
                            else:
                                if ct_sample_bgr is None:
                                    ct_sample_bgr = ct_sample.load_bgr()
                                img = imagelib.color_transfer (ct_mode, img, cv2.resize( ct_sample_bgr, (resolution,resolution), interpolation=cv2.INTER_LINEAR ) )


                        randomization_order = ['blur', 'noise', 'jpeg', 'down']
                        np.random.shuffle(randomization_order)
                        for random_distortion in randomization_order:
                            # Apply random blur
                            if random_distortion == 'blur' and random_blur:
                                blur_type = np.random.choice(['motion', 'gaussian'])

                                if blur_type == 'motion':
                                    blur_k = np.random.randint(10, 20)
                                    blur_angle = 360 * np.random.random()
                                    img = LinearMotionBlur(img, blur_k, blur_angle)
                                elif blur_type == 'gaussian':
                                    blur_sigma = 5 * np.random.random() + 3

                                    if blur_sigma < 5.0:
                                        kernel_size = 2.9 * blur_sigma  # 97% of weight
                                    else:
                                        kernel_size = 2.6 * blur_sigma  # 95% of weight
                                    kernel_size = int(kernel_size)
                                    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

                                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), blur_sigma)

                            # Apply random noise
                            if random_distortion == 'noise' and random_noise:
                                noise_type = np.random.choice(['gaussian', 'laplace', 'poisson'])
                                noise_scale = (20 * np.random.random() + 20)

                                if noise_type == 'gaussian':
                                    noise = np.random.normal(scale=noise_scale, size=img.shape)
                                    img += noise / 255.0
                                elif noise_type == 'laplace':
                                    noise = np.random.laplace(scale=noise_scale, size=img.shape)
                                    img += noise / 255.0
                                elif noise_type == 'poisson':
                                    noise_lam = (15 * np.random.random() + 15)
                                    noise = np.random.poisson(lam=noise_lam, size=img.shape)
                                    img += noise / 255.0

                            # Apply random jpeg compression
                            if random_distortion == 'jpeg' and random_jpeg:
                                img = np.clip(img*255, 0, 255).astype(np.uint8)
                                jpeg_compression_level = np.random.randint(50, 85)
                                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_compression_level]
                                _, enc_img = cv2.imencode('.jpg', img, encode_param)
                                img = cv2.imdecode(enc_img, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

                            # Apply random downsampling
                            if random_distortion == 'down' and random_downsample:
                                down_res = np.random.randint(int(0.125*resolution), int(0.25*resolution))
                                img = cv2.resize(img, (down_res, down_res), interpolation=cv2.INTER_CUBIC)
                                img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

                        if random_hsv_shift_amount != 0:
                            a = random_hsv_shift_amount
                            h_amount = max(1, int(360*a*0.5))
                            img_h, img_s, img_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                            img_h = (img_h + rnd_state.randint(-h_amount, h_amount+1) ) % 360
                            img_s = np.clip (img_s + (rnd_state.random()-0.5)*a, 0, 1 )
                            img_v = np.clip (img_v + (rnd_state.random()-0.5)*a, 0, 1 )
                            img = np.clip( cv2.cvtColor(cv2.merge([img_h, img_s, img_v]), cv2.COLOR_HSV2BGR) , 0, 1 )

                        # Apply random shadows
                        if isinstance(random_shadow, list):
                            shadow_opts = {}
                            for opt in random_shadow:
                                shadow_opts.update(opt)
                            if shadow_opts['enabled'] == True and sample_rnd_seed % 10 / 10 < 0.5:
                                high_ratio = (shadow_opts['high_bright_low'], shadow_opts['high_bright_high'])
                                low_ratio = (shadow_opts['shadow_low'], shadow_opts['shadow_high'])
                                img = shadow_highlights_augmentation(img, high_ratio=high_ratio, low_ratio=low_ratio, seed=sample_rnd_seed)
                        else:
                            if random_shadow == True and sample_rnd_seed % 10 / 10 < 0.5:
                                img = shadow_highlights_augmentation(img, seed=sample_rnd_seed)
                        img  = imagelib.warp_by_params (warp_params, img,  warp, transform, can_flip=True, border_replicate=border_replicate)
                        img = np.clip(img.astype(np.float32), 0, 1)

                        # Transform from BGR to desired channel_type
                        if channel_type == SPCT.BGR:
                            out_sample = img
                        elif channel_type == SPCT.LAB_RAND_TRANSFORM:
                            out_sample = random_lab_rotation(img, sample_rnd_seed)
                        elif channel_type == SPCT.G:
                            out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                        elif channel_type == SPCT.GGG:
                            out_sample = np.repeat ( np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),-1), (3,), -1)

                    # Final transformations
                    if nearest_resize_to is not None:
                        out_sample = cv2_resize(out_sample, (nearest_resize_to,nearest_resize_to), interpolation=cv2.INTER_NEAREST)

                    if not debug:
                        if normalize_tanh:
                            out_sample = np.clip (out_sample * 2.0 - 1.0, -1.0, 1.0)
                    if data_format == "NCHW":
                        out_sample = np.transpose(out_sample, (2,0,1) )
                elif sample_type == SPST.IMAGE:
                    img = sample_bgr
                    img  = imagelib.warp_by_params (warp_params, img,  warp, transform, can_flip=True, border_replicate=True)
                    img  = cv2.resize( img,  (resolution, resolution), interpolation=cv2.INTER_CUBIC )
                    out_sample = img

                    if data_format == "NCHW":
                        out_sample = np.transpose(out_sample, (2,0,1) )


                elif sample_type == SPST.LANDMARKS_ARRAY:
                    l = sample_landmarks
                    l = np.concatenate ( [ np.expand_dims(l[:,0] / w,-1), np.expand_dims(l[:,1] / h,-1) ], -1 )
                    l = np.clip(l, 0.0, 1.0)
                    out_sample = l
                elif sample_type == SPST.PITCH_YAW_ROLL or sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                    pitch,yaw,roll = sample.get_pitch_yaw_roll()
                    if warp_params['flip']:
                        yaw = -yaw

                    if sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                        pitch = np.clip( (pitch / math.pi) / 2.0 + 0.5, 0, 1)
                        yaw   = np.clip( (yaw / math.pi) / 2.0 + 0.5, 0, 1)
                        roll  = np.clip( (roll / math.pi) / 2.0 + 0.5, 0, 1)

                    out_sample = (pitch, yaw)
                else:
                    raise ValueError ('expected sample_type')

                outputs_sample.append ( out_sample )

            outputs += [outputs_sample]

        return outputs, warp_params['flip']
