import multiprocessing
import shutil

import cv2
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from DFLIMG import *
from facelib import FaceType, LandmarksProcessor


class FacesetResizerSubprocessor(Subprocessor):

    #override
    def __init__(self, image_paths, output_dirpath, image_size, face_type=None):
        self.image_paths = image_paths
        self.output_dirpath = output_dirpath
        self.image_size = image_size
        self.face_type = face_type
        self.result = []

        super().__init__('FacesetResizer', FacesetResizerSubprocessor.Cli, 600)

    #override
    def on_clients_initialized(self):
        io.progress_bar (None, len (self.image_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        base_dict = {'output_dirpath':self.output_dirpath, 'image_size':self.image_size, 'face_type':self.face_type}

        for device_idx in range( min(8, multiprocessing.cpu_count()) ):
            client_dict = base_dict.copy()
            device_name = f'CPU #{device_idx}'
            client_dict['device_name'] = device_name
            yield device_name, {}, client_dict

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

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.output_dirpath = client_dict['output_dirpath']
            self.image_size = client_dict['image_size']
            self.face_type = client_dict['face_type']
            self.log_info (f"Running on { client_dict['device_name'] }")

        #override
        def process_data(self, filepath):
            try:
                dflimg = DFLIMG.load (filepath)
                if dflimg is None or not dflimg.has_data():
                    self.log_err (f"{filepath.name} is not a dfl image file")
                else:
                    img = cv2_imread(filepath)
                    h,w = img.shape[:2]
                    if h != w:
                        raise Exception(f'w != h in {filepath}')
                    
                    image_size = self.image_size
                    face_type = self.face_type
                    output_filepath = self.output_dirpath / filepath.name
                    
                    if face_type is not None:
                        lmrks = dflimg.get_landmarks()
                        mat = LandmarksProcessor.get_transform_mat(lmrks, image_size, face_type)
                        
                        img = cv2.warpAffine(img, mat, (image_size, image_size), flags=cv2.INTER_LANCZOS4 )
                        img = np.clip(img, 0, 255).astype(np.uint8)
                        
                        cv2_imwrite ( str(output_filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )

                        dfl_dict = dflimg.get_dict()
                        dflimg = DFLIMG.load (output_filepath)
                        dflimg.set_dict(dfl_dict)
                        
                        xseg_mask = dflimg.get_xseg_mask()
                        if xseg_mask is not None:
                            xseg_res = 256
                            
                            xseg_lmrks = lmrks.copy()
                            xseg_lmrks *= (xseg_res / w)
                            xseg_mat = LandmarksProcessor.get_transform_mat(xseg_lmrks, xseg_res, face_type)
                            
                            xseg_mask = cv2.warpAffine(xseg_mask, xseg_mat, (xseg_res, xseg_res), flags=cv2.INTER_LANCZOS4 )
                            xseg_mask[xseg_mask < 0.5] = 0
                            xseg_mask[xseg_mask >= 0.5] = 1

                            dflimg.set_xseg_mask(xseg_mask)
                        
                        seg_ie_polys = dflimg.get_seg_ie_polys()
                        
                        for poly in seg_ie_polys.get_polys():
                            poly_pts = poly.get_pts()
                            poly_pts = LandmarksProcessor.transform_points(poly_pts, mat)
                            poly.set_points(poly_pts)
                            
                        dflimg.set_seg_ie_polys(seg_ie_polys)
                        
                        lmrks = LandmarksProcessor.transform_points(lmrks, mat)
                        dflimg.set_landmarks(lmrks)
    
                        image_to_face_mat = dflimg.get_image_to_face_mat()
                        if image_to_face_mat is not None:
                            image_to_face_mat = LandmarksProcessor.get_transform_mat ( dflimg.get_source_landmarks(), image_size, face_type )
                            dflimg.set_image_to_face_mat(image_to_face_mat)
                        dflimg.set_face_type( FaceType.toString(face_type) )
                        dflimg.save()
                        
                    else:
                        dfl_dict = dflimg.get_dict()
                         
                        scale = w / image_size
                        
                        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)                    
                        
                        cv2_imwrite ( str(output_filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )

                        dflimg = DFLIMG.load (output_filepath)
                        dflimg.set_dict(dfl_dict)
                        
                        lmrks = dflimg.get_landmarks()                    
                        lmrks /= scale
                        dflimg.set_landmarks(lmrks)
                        
                        seg_ie_polys = dflimg.get_seg_ie_polys()
                        seg_ie_polys.mult_points( 1.0 / scale)
                        dflimg.set_seg_ie_polys(seg_ie_polys)
                        
                        image_to_face_mat = dflimg.get_image_to_face_mat()
    
                        if image_to_face_mat is not None:
                            face_type = FaceType.fromString ( dflimg.get_face_type() )
                            image_to_face_mat = LandmarksProcessor.get_transform_mat ( dflimg.get_source_landmarks(), image_size, face_type )
                            dflimg.set_image_to_face_mat(image_to_face_mat)
                        dflimg.save()

                    return (1, filepath, output_filepath)
            except:
                self.log_err (f"Exception occured while processing file {filepath}. Error: {traceback.format_exc()}")

            return (0, filepath, None)

def process_folder ( dirpath):
    
    image_size = io.input_int(f"New image size", 512, valid_range=[128,2048])
    
    face_type = io.input_str ("Change face type", 'same', ['h','mf','f','wf','head','same']).lower()
    if face_type == 'same':
        face_type = None
    else:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]
                     

    output_dirpath = dirpath.parent / (dirpath.name + '_resized')
    output_dirpath.mkdir (exist_ok=True, parents=True)

    dirpath_parts = '/'.join( dirpath.parts[-2:])
    output_dirpath_parts = '/'.join( output_dirpath.parts[-2:] )
    io.log_info (f"Resizing faceset in {dirpath_parts}")
    io.log_info ( f"Processing to {output_dirpath_parts}")

    output_images_paths = pathex.get_image_paths(output_dirpath)
    if len(output_images_paths) > 0:
        for filename in output_images_paths:
            Path(filename).unlink()

    image_paths = [Path(x) for x in pathex.get_image_paths( dirpath )]
    result = FacesetResizerSubprocessor ( image_paths, output_dirpath, image_size, face_type).run()

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
