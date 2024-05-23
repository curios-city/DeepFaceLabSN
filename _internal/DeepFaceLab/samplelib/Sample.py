from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from core.cv2ex import *
from facelib import LandmarksProcessor
from core import imagelib
from core.imagelib import SegIEPolys

import zipfile

class SampleType(IntEnum):
    IMAGE = 0 #raw image

    FACE_BEGIN = 1
    FACE = 1                        #aligned face unsorted
    FACE_PERSON = 2                 #aligned face person
    FACE_TEMPORAL_SORTED = 3        #sorted by source filename
    FACE_END = 3

    QTY = 4

class Sample(object):
    __slots__ = ['sample_type',
                 'filename',
                 'face_type',
                 'shape',
                 'landmarks',
                 'seg_ie_polys',
                 'xseg_mask',
                 'xseg_mask_compressed',
                 'eyebrows_expand_mod',
                 'source_filename',
                 'person_name',
                 'pitch_yaw_roll',
                 '_filename_offset_size',
                ]

    def __init__(self, sample_type=None,
                       filename=None,
                       face_type=None,
                       shape=None,
                       landmarks=None,
                       seg_ie_polys=None,
                       xseg_mask=None,
                       xseg_mask_compressed=None,
                       eyebrows_expand_mod=None,
                       source_filename=None,
                       person_name=None,
                       pitch_yaw_roll=None,
                       **kwargs):

        self.sample_type = sample_type if sample_type is not None else SampleType.IMAGE
        self.filename = filename
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks) if landmarks is not None else None
        
        if isinstance(seg_ie_polys, SegIEPolys):
            self.seg_ie_polys = seg_ie_polys
        else:
            self.seg_ie_polys = SegIEPolys.load(seg_ie_polys)
        
        self.xseg_mask = xseg_mask
        self.xseg_mask_compressed = xseg_mask_compressed
        
        if self.xseg_mask_compressed is None and self.xseg_mask is not None:
            xseg_mask = np.clip( imagelib.normalize_channels(xseg_mask, 1)*255, 0, 255 ).astype(np.uint8)        
            ret, xseg_mask_compressed = cv2.imencode('.png', xseg_mask)
            if not ret:
                raise Exception("Sample(): unable to generate xseg_mask_compressed")
            self.xseg_mask_compressed = xseg_mask_compressed
            self.xseg_mask = None
 
        self.eyebrows_expand_mod = eyebrows_expand_mod if eyebrows_expand_mod is not None else 1.0
        self.source_filename = source_filename
        self.person_name = person_name
        self.pitch_yaw_roll = pitch_yaw_roll

        self._filename_offset_size = None

    def has_xseg_mask(self):
        return self.xseg_mask is not None or self.xseg_mask_compressed is not None
        
    def get_xseg_mask(self):
        if self.xseg_mask_compressed is not None:
            xseg_mask = cv2.imdecode(self.xseg_mask_compressed, cv2.IMREAD_UNCHANGED)
            if len(xseg_mask.shape) == 2:
                xseg_mask = xseg_mask[...,None]
            return xseg_mask.astype(np.float32) / 255.0
        return self.xseg_mask
        
    def get_pitch_yaw_roll(self):
        if self.pitch_yaw_roll is None:
            self.pitch_yaw_roll = LandmarksProcessor.estimate_pitch_yaw_roll(self.landmarks, size=self.shape[1])
        return self.pitch_yaw_roll

    def set_filename_offset_size(self, filename, offset, size):
        self._filename_offset_size = (filename, offset, size)

    def read_raw_file(self, filename=None):
        if self._filename_offset_size is not None:
            filename, offset, size = self._filename_offset_size
            if filename.endswith(".zip"):
                with zipfile.ZipFile(filename, 'r') as zipObj:
                    return zipObj.read(self.filename)
            else:
                with open(filename, "rb") as f:
                    f.seek( offset, 0)
                    return f.read (size)
        else:
            with open(filename, "rb") as f:
                return f.read()

    def load_bgr(self):
        img = cv2_imread (self.filename, loader_func=self.read_raw_file).astype(np.float32) / 255.0
        return img

    def get_config(self):
        return {'sample_type': self.sample_type,
                'filename': self.filename,
                'face_type': self.face_type,
                'shape': self.shape,
                'landmarks': self.landmarks.tolist(),
                'seg_ie_polys': self.seg_ie_polys.dump(),
                'xseg_mask' : self.xseg_mask,
                'xseg_mask_compressed' : self.xseg_mask_compressed,
                'eyebrows_expand_mod': self.eyebrows_expand_mod,
                'source_filename': self.source_filename,
                'person_name': self.person_name
               }
