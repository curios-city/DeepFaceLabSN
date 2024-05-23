from enum import IntEnum
from pathlib import Path
import multiprocessing

import cv2
import numpy as np
import math

#from core.cv2ex import *
#from facelib import LandmarksProcessor
#from core import imagelib
from UI.utils.DFLIMG.SegIEPolys import SegIEPolys
from UI.utils.DFLIMG.SubprocessorBase import Subprocessor






def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")

    if c == 0 and target_channels > 0:
        img = img[...,np.newaxis]
        c = 1

    if c == 1 and target_channels > 1:
        img = np.repeat (img, target_channels, -1)
        c = target_channels

    if c > target_channels:
        img = img[...,0:target_channels]
        c = target_channels

    return img


from enum import IntEnum

class FaceType(IntEnum):
    #enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20

    MARK_ONLY = 100, #no align at all, just embedded faceinfo

    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_dict[face_type]

to_string_dict = { FaceType.HALF : 'half_face',
                   FaceType.MID_FULL : 'midfull_face',
                   FaceType.FULL : 'full_face',
                   FaceType.FULL_NO_ALIGN : 'full_face_no_align',
                   FaceType.WHOLE_FACE : 'whole_face',
                   FaceType.HEAD : 'head',
                   FaceType.HEAD_NO_ALIGN : 'head_no_align',
                   
                   FaceType.MARK_ONLY :'mark_only',  
                 }

from_string_dict = { to_string_dict[x] : x for x in to_string_dict.keys() }  

landmarks_68_3D = np.array( [
[-73.393523  , -29.801432   , 47.667532   ], #00
[-72.775014  , -10.949766   , 45.909403   ], #01
[-70.533638  , 7.929818     , 44.842580   ], #02
[-66.850058  , 26.074280    , 43.141114   ], #03
[-59.790187  , 42.564390    , 38.635298   ], #04
[-48.368973  , 56.481080    , 30.750622   ], #05
[-34.121101  , 67.246992    , 18.456453   ], #06
[-17.875411  , 75.056892    , 3.609035    ], #07
[0.098749    , 77.061286    , -0.881698   ], #08
[17.477031   , 74.758448    , 5.181201    ], #09
[32.648966   , 66.929021    , 19.176563   ], #10
[46.372358   , 56.311389    , 30.770570   ], #11
[57.343480   , 42.419126    , 37.628629   ], #12
[64.388482   , 25.455880    , 40.886309   ], #13
[68.212038   , 6.990805     , 42.281449   ], #14
[70.486405   , -11.666193   , 44.142567   ], #15
[71.375822   , -30.365191   , 47.140426   ], #16
[-61.119406  , -49.361602   , 14.254422   ], #17
[-51.287588  , -58.769795   , 7.268147    ], #18
[-37.804800  , -61.996155   , 0.442051    ], #19
[-24.022754  , -61.033399   , -6.606501   ], #20
[-11.635713  , -56.686759   , -11.967398  ], #21
[12.056636   , -57.391033   , -12.051204  ], #22
[25.106256   , -61.902186   , -7.315098   ], #23
[38.338588   , -62.777713   , -1.022953   ], #24
[51.191007   , -59.302347   , 5.349435    ], #25
[60.053851   , -50.190255   , 11.615746   ], #26
[0.653940    , -42.193790   , -13.380835  ], #27
[0.804809    , -30.993721   , -21.150853  ], #28
[0.992204    , -19.944596   , -29.284036  ], #29
[1.226783    , -8.414541    , -36.948060  ], #00
[-14.772472  , 2.598255     , -20.132003  ], #01
[-7.180239   , 4.751589     , -23.536684  ], #02
[0.555920    , 6.562900     , -25.944448  ], #03
[8.272499    , 4.661005     , -23.695741  ], #04
[15.214351   , 2.643046     , -20.858157  ], #05
[-46.047290  , -37.471411   , 7.037989    ], #06
[-37.674688  , -42.730510   , 3.021217    ], #07
[-27.883856  , -42.711517   , 1.353629    ], #08
[-19.648268  , -36.754742   , -0.111088   ], #09
[-28.272965  , -35.134493   , -0.147273   ], #10
[-38.082418  , -34.919043   , 1.476612    ], #11
[19.265868   , -37.032306   , -0.665746   ], #12
[27.894191   , -43.342445   , 0.247660    ], #13
[37.437529   , -43.110822   , 1.696435    ], #14
[45.170805   , -38.086515   , 4.894163    ], #15
[38.196454   , -35.532024   , 0.282961    ], #16
[28.764989   , -35.484289   , -1.172675   ], #17
[-28.916267  , 28.612716    , -2.240310   ], #18
[-17.533194  , 22.172187    , -15.934335  ], #19
[-6.684590   , 19.029051    , -22.611355  ], #20
[0.381001    , 20.721118    , -23.748437  ], #21
[8.375443    , 19.035460    , -22.721995  ], #22
[18.876618   , 22.394109    , -15.610679  ], #23
[28.794412   , 28.079924    , -3.217393   ], #24
[19.057574   , 36.298248    , -14.987997  ], #25
[8.956375    , 39.634575    , -22.554245  ], #26
[0.381549    , 40.395647    , -23.591626  ], #27
[-7.428895   , 39.836405    , -22.406106  ], #28
[-18.160634  , 36.677899    , -15.121907  ], #29
[-24.377490  , 28.677771    , -4.785684   ], #30
[-6.897633   , 25.475976    , -20.893742  ], #31
[0.340663    , 26.014269    , -22.220479  ], #32
[8.444722    , 25.326198    , -21.025520  ], #33
[24.474473   , 28.323008    , -5.712776   ], #34
[8.449166    , 30.596216    , -20.671489  ], #35
[0.205322    , 31.408738    , -21.903670  ], #36 
[-7.198266   , 30.844876    , -20.328022  ]  #37
], dtype=np.float32)

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def estimate_pitch_yaw_roll(aligned_landmarks, size=256):
    """
    returns pitch,yaw,roll [-pi/2...+pi/2]
    """
    shape = (size,size)
    focal_length = shape[1]
    camera_center = (shape[1] / 2, shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype=np.float32)

    (_, rotation_vector, _) = cv2.solvePnP(
        np.concatenate( (landmarks_68_3D[:27],   landmarks_68_3D[30:36]) , axis=0) ,
        np.concatenate( (aligned_landmarks[:27], aligned_landmarks[30:36]) , axis=0).astype(np.float32),
        camera_matrix,
        np.zeros((4, 1)) )

    pitch, yaw, roll = rotationMatrixToEulerAngles( cv2.Rodrigues(rotation_vector)[0] )
   
    half_pi = math.pi / 2.0
    pitch = np.clip ( pitch, -half_pi, half_pi )
    yaw   = np.clip ( yaw ,  -half_pi, half_pi )
    roll  = np.clip ( roll,  -half_pi, half_pi )

    return -pitch, yaw, roll

def DFLIMGload(filepath, loader_func=None):
        from UI.utils.DFLIMG.DFLJPG import DFLJPG
        if filepath.suffix == '.jpg':
            return DFLJPG.load ( str(filepath), loader_func=loader_func )
        else:
            return None

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
            xseg_mask = np.clip( normalize_channels(xseg_mask, 1)*255, 0, 255 ).astype(np.uint8)        
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
            self.pitch_yaw_roll = estimate_pitch_yaw_roll(self.landmarks, size=self.shape[1])
        return self.pitch_yaw_roll

    def set_filename_offset_size(self, filename, offset, size):
        self._filename_offset_size = (filename, offset, size)

    def read_raw_file(self, filename=None):
        if self._filename_offset_size is not None:
            filename, offset, size = self._filename_offset_size
            with open(filename, "rb") as f:
                f.seek( offset, 0)
                return f.read (size)
        else:
            with open(filename, "rb") as f:
                return f.read()

    def load_bgr(self):
        from UI.utils.DFLIMG.DFLJPG import cv2_imread
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


def load_face_samples ( image_paths):
        result = FaceSamplesLoaderSubprocessor(image_paths).run()
        sample_list = []

        for filename, data in result:
            if data is None:
                continue
            ( face_type,
              shape,
              landmarks,
              seg_ie_polys,
              xseg_mask_compressed,
              eyebrows_expand_mod,
              source_filename ) = data
              
            sample_list.append( Sample(filename=filename,
                                        sample_type=SampleType.FACE,
                                        face_type=FaceType.fromString (face_type),
                                        shape=shape,
                                        landmarks=landmarks,
                                        seg_ie_polys=seg_ie_polys,
                                        xseg_mask_compressed=xseg_mask_compressed,
                                        eyebrows_expand_mod=eyebrows_expand_mod,
                                        source_filename=source_filename,
                                    ))
        return sample_list

class FaceSamplesLoaderSubprocessor(Subprocessor):
    #override
    def __init__(self, image_paths ):
        self.image_paths = image_paths
        self.image_paths_len = len(image_paths)
        self.idxs = [*range(self.image_paths_len)]
        self.result = [None]*self.image_paths_len
        super().__init__('FaceSamplesLoader', FaceSamplesLoaderSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        pass

    #override
    def on_clients_finalized(self):
        pass

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {}

    #override
    def get_data(self, host_dict):
        if len (self.idxs) > 0:
            idx = self.idxs.pop(0)
            return idx, self.image_paths[idx]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.idxs.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        idx, dflimg = result
        self.result[idx] = (self.image_paths[idx], dflimg)

    #override
    def get_result(self):
        return self.result
    

    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            idx, filename = data
            dflimg = DFLIMGload (Path(filename))

            if dflimg is None or not dflimg.has_data():
                self.log_err (f"FaceSamplesLoader: {filename} is not a dfl image file.")
                data = None
            else:
                data = (dflimg.get_face_type(),
                        dflimg.get_shape(),
                        dflimg.get_landmarks(),
                        dflimg.get_seg_ie_polys(),
                        dflimg.get_xseg_mask_compressed(),
                        dflimg.get_eyebrows_expand_mod(),
                        dflimg.get_source_filename() )

            return idx, data

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[1]
