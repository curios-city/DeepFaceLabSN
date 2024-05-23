import cv2
from facelib import FaceType, LandmarksProcessor
from core import imagelib
import datetime
import json
import os
from pathlib import Path
import shutil
import numpy as np

from samplelib.Sample import Sample

def prepare_sample(sample: Sample, options: dict, resolution: int, face_type_enum: int):
    sample_bgr = sample.load_bgr()
    sample_mask, sample_mask_em = get_masks(sample, sample_bgr, sample.landmarks, eye_prio=options['eyes_prio'], mouth_prio=options['mouth_prio'])

    sample_bgr = get_input_image(sample_bgr, sample.face_type, sample.landmarks, resolution, face_type_enum)
    sample_mask = get_input_image(sample_mask, sample.face_type, sample.landmarks, resolution, face_type_enum)
    sample_mask_em = get_input_image(sample_mask_em, sample.face_type, sample.landmarks, resolution, face_type_enum)

    return sample_bgr, sample_mask, sample_mask_em

def get_full_face_mask(sample: Sample, sample_bgr, sample_landmarks):
    xseg_mask = sample.get_xseg_mask()
    if xseg_mask is not None:
        if xseg_mask.shape[0] != sample_bgr.shape[0] or xseg_mask.shape[1] != sample_bgr.shape[1]:
            xseg_mask = cv2.resize(
                xseg_mask, (sample_bgr.shape[0], sample_bgr.shape[1]), interpolation=cv2.INTER_CUBIC)
            xseg_mask = imagelib.normalize_channels(xseg_mask, 1)
        return np.clip(xseg_mask, 0, 1)
    else:
        full_face_mask = LandmarksProcessor.get_image_hull_mask(
            sample_bgr.shape, sample_landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod)
        return np.clip(full_face_mask, 0, 1)

def get_eyes_mask(sample_bgr, sample_landmarks):
    eyes_mask = LandmarksProcessor.get_image_eye_mask(sample_bgr.shape, sample_landmarks)
    # set eye masks to 1-2
    clip = np.clip(eyes_mask, 0, 1)
    clip[clip > 0.1] += 1
    return clip

def get_mouth_mask(sample_bgr, sample_landmarks):
    mouth_mask = LandmarksProcessor.get_image_mouth_mask(sample_bgr.shape, sample_landmarks)
    # set eye masks to 2-3
    clip = np.clip(mouth_mask, 0, 1)
    clip[clip > 0.1] += 2
    return clip

def get_eyes_mouth_mask():                
    eyes_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample_landmarks)
    mouth_mask = LandmarksProcessor.get_image_mouth_mask (sample_bgr.shape, sample_landmarks)
    mask = eyes_mask + mouth_mask
    return np.clip(mask, 0, 1)

def get_full_face_eyes(sample, sample_bgr, sample_landmarks):
    # sets both eyes and mouth mask parts
    img = get_full_face_mask(sample, sample_bgr, sample_landmarks)
    mask = img.copy()
    mask[mask != 0.0] = 1.0
    eye_mask = get_eyes_mask(sample_bgr, sample_landmarks) * mask
    img = np.where(eye_mask > 1, eye_mask, img)
    mouth_mask = get_mouth_mask(sample_bgr, sample_landmarks) * mask
    img = np.where(mouth_mask > 2, mouth_mask, img)

def get_masks(sample, sample_bgr, sample_landmarks, eye_prio = False, mouth_prio = False):
    mask = get_full_face_mask(sample, sample_bgr, sample_landmarks)

    mask_em = mask.copy()
    if eye_prio or mouth_prio:
        mask_em[mask_em != 0.0] = 1.0
        eye_mask = get_eyes_mask(
            sample_bgr, sample_landmarks) * mask_em
        mask = np.where(eye_mask > 1, eye_mask, mask)
        mouth_mask = get_mouth_mask(
            sample_bgr, sample_landmarks) * mask_em
        mask = np.where(mouth_mask > 2, mouth_mask, mask)

    return mask, mask_em

def get_input_image(image, sample_face_type, sample_landmarks, resolution, face_type):
    if face_type != sample_face_type and sample_face_type != FaceType.CUSTOM: # custom always valid for stuff like for wf custom equivalent
        mat = LandmarksProcessor.get_transform_mat(sample_landmarks, resolution, face_type)
        image = cv2.warpAffine(image, mat, (resolution, resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR )
    else:
        if image.shape[0] != resolution:
            image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LINEAR )
    return image

def data_format_change(image):
    if len(image.shape) == 2:
        image = image[..., None]
    image = np.transpose(image, (2,0,1) )
    image = np.expand_dims(image, axis=0)

    return image

def print_sample_status(sample):
    print (sample.shape)
    print (np.amax(sample))