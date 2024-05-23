import traceback
import math
import multiprocessing
import operator
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

import facelib
from core import imagelib
from core import mathlib
from facelib import FaceType, LandmarksProcessor
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from core import pathex
from core.cv2ex import *
from DFLIMG import *

DEBUG = False

class ExtractSubprocessor(Subprocessor):
    class Data(object):
        def __init__(self, filepath=None, rects=None, landmarks = None, landmarks_accurate=True, manual=False, force_output_path=None, final_output_files = None):
            self.filepath = filepath
            self.rects = rects or []
            self.rects_rotation = 0
            self.landmarks_accurate = landmarks_accurate
            self.manual = manual
            self.landmarks = landmarks or []
            self.force_output_path = force_output_path
            self.final_output_files = final_output_files or []
            self.faces_detected = 0

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.type                 = client_dict['type']
            self.image_size           = client_dict['image_size']
            self.jpeg_quality         = client_dict['jpeg_quality']
            self.face_type            = client_dict['face_type']
            self.max_faces_from_image = client_dict['max_faces_from_image']
            self.device_idx           = client_dict['device_idx']
            self.cpu_only             = client_dict['device_type'] == 'CPU'
            self.final_output_path    = client_dict['final_output_path']
            self.output_debug_path    = client_dict['output_debug_path']

            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None and DEBUG:
                sys.stdin = os.fdopen(stdin_fd)

            if self.cpu_only:
                device_config = nn.DeviceConfig.CPU()
                place_model_on_cpu = True
            else:
                device_config = nn.DeviceConfig.GPUIndexes ([self.device_idx])
                place_model_on_cpu = device_config.devices[0].total_mem_gb < 4

            if self.type == 'all' or 'rects' in self.type or 'landmarks' in self.type:
                nn.initialize (device_config)

            self.log_info (f"Running on {client_dict['device_name'] }")

            if self.type == 'all' or self.type == 'rects-s3fd' or 'landmarks' in self.type:
                self.rects_extractor = facelib.S3FDExtractor(place_model_on_cpu=place_model_on_cpu)

            if self.type == 'all' or 'landmarks' in self.type:
                # for head type, extract "3D landmarks"
                self.landmarks_extractor = facelib.FANExtractor(landmarks_3D=self.face_type >= FaceType.HEAD,
                                                                place_model_on_cpu=place_model_on_cpu)

            self.cached_image = (None, None)

        #override
        def process_data(self, data):
            if 'landmarks' in self.type and len(data.rects) == 0:
                return data

            filepath = data.filepath
            cached_filepath, image = self.cached_image
            if cached_filepath != filepath:
                image = cv2_imread( filepath )
                if image is None:
                    self.log_err (f'Failed to open {filepath}, reason: cv2_imread() fail.')
                    return data
                image = imagelib.normalize_channels(image, 3)
                image = imagelib.cut_odd_image(image)
                self.cached_image = ( filepath, image )

            h, w, c = image.shape

            if 'rects' in self.type or self.type == 'all':
                data = ExtractSubprocessor.Cli.rects_stage (data=data,
                                                            image=image,
                                                            max_faces_from_image=self.max_faces_from_image,
                                                            rects_extractor=self.rects_extractor,
                                                            )

            if 'landmarks' in self.type or self.type == 'all':
                data = ExtractSubprocessor.Cli.landmarks_stage (data=data,
                                                                image=image,
                                                                landmarks_extractor=self.landmarks_extractor,
                                                                rects_extractor=self.rects_extractor,
                                                                )

            if self.type == 'final' or self.type == 'all':
                data = ExtractSubprocessor.Cli.final_stage(data=data,
                                                           image=image,
                                                           face_type=self.face_type,
                                                           image_size=self.image_size,
                                                           jpeg_quality=self.jpeg_quality,
                                                           output_debug_path=self.output_debug_path,
                                                           final_output_path=self.final_output_path,
                                                           )
            return data

        @staticmethod
        def rects_stage(data,
                        image,
                        max_faces_from_image,
                        rects_extractor,
                        ):
            h,w,c = image.shape
            if min(h,w) < 128:
                # Image is too small
                data.rects = []
            else:
                for rot in ([0, 90, 270, 180]):
                    if rot == 0:
                        rotated_image = image
                    elif rot == 90:
                        rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
                    elif rot == 180:
                        rotated_image = image[::-1,::-1,:]
                    elif rot == 270:
                        rotated_image = image.swapaxes( 0,1 )[::-1,:,:]
                    rects = data.rects = rects_extractor.extract (rotated_image, is_bgr=True)
                    if len(rects) != 0:
                        data.rects_rotation = rot
                        break
                if max_faces_from_image is not None and \
                   max_faces_from_image > 0 and \
                   len(data.rects) > 0:
                    data.rects = data.rects[0:max_faces_from_image]
            return data


        @staticmethod
        def landmarks_stage(data,
                            image,
                            landmarks_extractor,
                            rects_extractor,
                            ):
            h, w, ch = image.shape

            if data.rects_rotation == 0:
                rotated_image = image
            elif data.rects_rotation == 90:
                rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
            elif data.rects_rotation == 180:
                rotated_image = image[::-1,::-1,:]
            elif data.rects_rotation == 270:
                rotated_image = image.swapaxes( 0,1 )[::-1,:,:]

            data.landmarks = landmarks_extractor.extract (rotated_image, data.rects, rects_extractor if (data.landmarks_accurate) else None, is_bgr=True)
            if data.rects_rotation != 0:
                for i, (rect, lmrks) in enumerate(zip(data.rects, data.landmarks)):
                    new_rect, new_lmrks = rect, lmrks
                    (l,t,r,b) = rect
                    if data.rects_rotation == 90:
                        new_rect = ( t, h-l, b, h-r)
                        if lmrks is not None:
                            new_lmrks = lmrks[:,::-1].copy()
                            new_lmrks[:,1] = h - new_lmrks[:,1]
                    elif data.rects_rotation == 180:
                        if lmrks is not None:
                            new_rect = ( w-l, h-t, w-r, h-b)
                            new_lmrks = lmrks.copy()
                            new_lmrks[:,0] = w - new_lmrks[:,0]
                            new_lmrks[:,1] = h - new_lmrks[:,1]
                    elif data.rects_rotation == 270:
                        new_rect = ( w-b, l, w-t, r )
                        if lmrks is not None:
                            new_lmrks = lmrks[:,::-1].copy()
                            new_lmrks[:,0] = w - new_lmrks[:,0]
                    data.rects[i], data.landmarks[i] = new_rect, new_lmrks

            return data

        @staticmethod
        def final_stage(data,
                        image,
                        face_type,
                        image_size,
                        jpeg_quality,
                        output_debug_path=None,
                        final_output_path=None,
                        ):
            data.final_output_files = []
            filepath = data.filepath
            rects = data.rects
            landmarks = data.landmarks

            if output_debug_path is not None:
                debug_image = image.copy()

            face_idx = 0
            for rect, image_landmarks in zip( rects, landmarks ):
                if image_landmarks is None:
                    continue

                rect = np.array(rect)

                if face_type == FaceType.MARK_ONLY:
                    image_to_face_mat = None
                    face_image = image
                    face_image_landmarks = image_landmarks
                else:
                    image_to_face_mat = LandmarksProcessor.get_transform_mat (image_landmarks, image_size, face_type)

                    face_image = cv2.warpAffine(image, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)
                    face_image_landmarks = LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat)

                    landmarks_bbox = LandmarksProcessor.transform_points ( [ (0,0), (0,image_size-1), (image_size-1, image_size-1), (image_size-1,0) ], image_to_face_mat, True)

                    rect_area      = mathlib.polygon_area(np.array(rect[[0,2,2,0]]).astype(np.float32), np.array(rect[[1,1,3,3]]).astype(np.float32))
                    landmarks_area = mathlib.polygon_area(landmarks_bbox[:,0].astype(np.float32), landmarks_bbox[:,1].astype(np.float32) )

                    if not data.manual and face_type <= FaceType.FULL_NO_ALIGN and landmarks_area > 4*rect_area: #get rid of faces which umeyama-landmark-area > 4*detector-rect-area
                        continue

                    if output_debug_path is not None:
                        LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, face_type, image_size, transparent_mask=True)

                output_path = final_output_path
                if data.force_output_path is not None:
                    output_path = data.force_output_path

                output_filepath = output_path / f"{filepath.stem}_{face_idx}.jpg"
                cv2_imwrite(output_filepath, face_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality ] )

                dflimg = DFLJPG.load(output_filepath)
                dflimg.set_face_type(FaceType.toString(face_type))
                dflimg.set_landmarks(face_image_landmarks.tolist())
                dflimg.set_source_filename(filepath.name)
                dflimg.set_source_rect(rect)
                dflimg.set_source_landmarks(image_landmarks.tolist())
                dflimg.set_image_to_face_mat(image_to_face_mat)
                dflimg.save()

                data.final_output_files.append (output_filepath)
                face_idx += 1
            data.faces_detected = face_idx

            if output_debug_path is not None:
                cv2_imwrite( output_debug_path / (filepath.stem+'.jpg'), debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

            return data

        #overridable
        def get_data_name (self, data):
            #return string identificator of your data
            return data.filepath

    @staticmethod
    def get_devices_for_config (type, device_config):
        devices = device_config.devices
        cpu_only = len(devices) == 0

        if 'rects'     in type or \
           'landmarks' in type or \
           'all'       in type:

            if not cpu_only:
                if type == 'landmarks-manual':
                    devices = [devices.get_best_device()]

                result = []

                for device in devices:
                    count = 1

                    if count == 1:
                        result += [ (device.index, 'GPU', device.name, device.total_mem_gb) ]
                    else:
                        for i in range(count):
                            result += [ (device.index, 'GPU', f"{device.name} #{i}", device.total_mem_gb) ]

                return result
            else:
                if type == 'landmarks-manual':
                    return [ (0, 'CPU', 'CPU', 0 ) ]
                else:
                    return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in range( min(8, multiprocessing.cpu_count() // 2) ) ]

        elif type == 'final':
            return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in (range(min(8, multiprocessing.cpu_count())) if not DEBUG else [0]) ]

    def __init__(self, input_data, type, image_size=None, jpeg_quality=None, face_type=None, output_debug_path=None, manual_window_size=0, max_faces_from_image=0, final_output_path=None, device_config=None):
        if type == 'landmarks-manual':
            for x in input_data:
                x.manual = True

        self.input_data = input_data

        self.type = type
        self.image_size = image_size
        self.jpeg_quality = jpeg_quality
        self.face_type = face_type
        self.output_debug_path = output_debug_path
        self.final_output_path = final_output_path
        self.manual_window_size = manual_window_size
        self.max_faces_from_image = max_faces_from_image
        self.result = []

        self.devices = ExtractSubprocessor.get_devices_for_config(self.type, device_config)

        super().__init__('Extractor', ExtractSubprocessor.Cli,
                             999999 if type == 'landmarks-manual' or DEBUG else 120)

    #override
    def on_clients_initialized(self):
        if self.type == 'landmarks-manual':
            self.wnd_name = 'Manual pass'
            io.named_window(self.wnd_name)
            io.capture_mouse(self.wnd_name)
            io.capture_keys(self.wnd_name)

            self.cache_original_image = (None, None)
            self.cache_image = (None, None)
            self.cache_text_lines_img = (None, None)
            self.hide_help = False
            self.landmarks_accurate = True
            self.force_landmarks = False

            self.landmarks = None
            self.x = 0
            self.y = 0
            self.rect_size = 100
            self.rect_locked = False
            self.extract_needed = True

            self.image = None
            self.image_filepath = None

        io.progress_bar (None, len (self.input_data))

    #override
    def on_clients_finalized(self):
        if self.type == 'landmarks-manual':
            io.destroy_all_windows()

        io.progress_bar_close()

    #override
    def process_info_generator(self):
        base_dict = {'type' : self.type,
                     'image_size': self.image_size,
                     'jpeg_quality' : self.jpeg_quality,
                     'face_type': self.face_type,
                     'max_faces_from_image':self.max_faces_from_image,
                     'output_debug_path': self.output_debug_path,
                     'final_output_path': self.final_output_path,
                     'stdin_fd': sys.stdin.fileno() }


        for (device_idx, device_type, device_name, device_total_vram_gb) in self.devices:
            client_dict = base_dict.copy()
            client_dict['device_idx'] = device_idx
            client_dict['device_name'] = device_name
            client_dict['device_type'] = device_type
            yield client_dict['device_name'], {}, client_dict

    #override
    def get_data(self, host_dict):
        if self.type == 'landmarks-manual':
            need_remark_face = False
            while len (self.input_data) > 0:
                data = self.input_data[0]
                filepath, data_rects, data_landmarks = data.filepath, data.rects, data.landmarks
                is_frame_done = False

                if self.image_filepath != filepath:
                    self.image_filepath = filepath
                    if self.cache_original_image[0] == filepath:
                        self.original_image = self.cache_original_image[1]
                    else:
                        self.original_image = imagelib.normalize_channels( cv2_imread( filepath ), 3 )

                        self.cache_original_image = (filepath, self.original_image )

                    (h,w,c) = self.original_image.shape
                    self.view_scale = 1.0 if self.manual_window_size == 0 else self.manual_window_size / ( h * (16.0/9.0) )

                    if self.cache_image[0] == (h,w,c) + (self.view_scale,filepath):
                        self.image = self.cache_image[1]
                    else:
                        self.image = cv2.resize (self.original_image, ( int(w*self.view_scale), int(h*self.view_scale) ), interpolation=cv2.INTER_LINEAR)
                        self.cache_image = ( (h,w,c) + (self.view_scale,filepath), self.image )

                    (h,w,c) = self.image.shape

                    sh = (0,0, w, min(100, h) )
                    if self.cache_text_lines_img[0] == sh:
                        self.text_lines_img = self.cache_text_lines_img[1]
                    else:
                        self.text_lines_img = (imagelib.get_draw_text_lines ( self.image, sh,
                                                        [   '[L Mouse click] - lock/unlock selection. [Mouse wheel] - change rect',
                                                            '[R Mouse Click] - manual face rectangle',
                                                            '[Enter] / [Space] - confirm / skip frame',
                                                            '[,] [.]- prev frame, next frame. [Q] - skip remaining frames',
                                                            '[a] - accuracy on/off (more fps)',
                                                            '[h] - hide this help'
                                                        ], (1, 1, 1) )*255).astype(np.uint8)

                        self.cache_text_lines_img = (sh, self.text_lines_img)

                if need_remark_face: # need remark image from input data that already has a marked face?
                    need_remark_face = False
                    if len(data_rects) != 0: # If there was already a face then lock the rectangle to it until the mouse is clicked
                        self.rect = data_rects.pop()
                        self.landmarks = data_landmarks.pop()
                        data_rects.clear()
                        data_landmarks.clear()

                        self.rect_locked = True
                        self.rect_size = ( self.rect[2] - self.rect[0] ) / 2
                        self.x = ( self.rect[0] + self.rect[2] ) / 2
                        self.y = ( self.rect[1] + self.rect[3] ) / 2
                        self.redraw()

                if len(data_rects) == 0:
                    (h,w,c) = self.image.shape
                    while True:
                        io.process_messages(0.0001)

                        if not self.force_landmarks:
                            new_x = self.x
                            new_y = self.y

                        new_rect_size = self.rect_size

                        mouse_events = io.get_mouse_events(self.wnd_name)
                        for ev in mouse_events:
                            (x, y, ev, flags) = ev
                            if ev == io.EVENT_MOUSEWHEEL and not self.rect_locked:
                                mod = 1 if flags > 0 else -1
                                diff = 1 if new_rect_size <= 40 else np.clip(new_rect_size / 10, 1, 10)
                                new_rect_size = max (5, new_rect_size + diff*mod)
                            elif ev == io.EVENT_LBUTTONDOWN:
                                if self.force_landmarks:
                                    self.x = new_x
                                    self.y = new_y
                                    self.force_landmarks = False
                                    self.rect_locked = True
                                    self.redraw()
                                else:
                                    self.rect_locked = not self.rect_locked
                                    self.extract_needed = True
                            elif ev == io.EVENT_RBUTTONDOWN:
                                self.force_landmarks = not self.force_landmarks
                                if self.force_landmarks:
                                    self.rect_locked = False
                            elif not self.rect_locked:
                                new_x = np.clip (x, 0, w-1) / self.view_scale
                                new_y = np.clip (y, 0, h-1) / self.view_scale

                        key_events = io.get_key_events(self.wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

                        if key == ord('\r') or key == ord('\n'):
                            #confirm frame
                            is_frame_done = True
                            data_rects.append (self.rect)
                            data_landmarks.append (self.landmarks)
                            break
                        elif key == ord(' '):
                            #confirm skip frame
                            is_frame_done = True
                            break
                        elif key == ord(',')  and len(self.result) > 0:
                            #go prev frame

                            if self.rect_locked:
                                self.rect_locked = False
                                # Only save the face if the rect is still locked
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)


                            self.input_data.insert(0, self.result.pop() )
                            io.progress_bar_inc(-1)
                            need_remark_face = True

                            break
                        elif key == ord('.'):
                            #go next frame

                            if self.rect_locked:
                                self.rect_locked = False
                                # Only save the face if the rect is still locked
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)

                            need_remark_face = True
                            is_frame_done = True
                            break
                        elif key == ord('q'):
                            #skip remaining

                            if self.rect_locked:
                                self.rect_locked = False
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)

                            while len(self.input_data) > 0:
                                self.result.append( self.input_data.pop(0) )
                                io.progress_bar_inc(1)

                            break

                        elif key == ord('h'):
                            self.hide_help = not self.hide_help
                            break
                        elif key == ord('a'):
                            self.landmarks_accurate = not self.landmarks_accurate
                            break

                        if self.force_landmarks:
                            pt2 = np.float32([new_x, new_y])
                            pt1 = np.float32([self.x, self.y])

                            pt_vec_len = npla.norm(pt2-pt1)
                            pt_vec = pt2-pt1
                            if pt_vec_len != 0:
                                pt_vec /= pt_vec_len

                            self.rect_size = pt_vec_len
                            self.rect = ( int(self.x-self.rect_size),
                                          int(self.y-self.rect_size),
                                          int(self.x+self.rect_size),
                                          int(self.y+self.rect_size) )

                            if pt_vec_len > 0:
                                lmrks = np.concatenate ( (np.zeros ((17,2), np.float32), LandmarksProcessor.landmarks_2D), axis=0 )
                                lmrks -= lmrks[30:31,:]
                                mat = cv2.getRotationMatrix2D( (0, 0), -np.arctan2( pt_vec[1], pt_vec[0] )*180/math.pi , pt_vec_len)
                                mat[:, 2] += (self.x, self.y)
                                self.landmarks = LandmarksProcessor.transform_points(lmrks, mat )


                            self.redraw()

                        elif self.x != new_x or \
                           self.y != new_y or \
                           self.rect_size != new_rect_size or \
                           self.extract_needed:
                            self.x = new_x
                            self.y = new_y
                            self.rect_size = new_rect_size
                            self.rect = ( int(self.x-self.rect_size),
                                          int(self.y-self.rect_size),
                                          int(self.x+self.rect_size),
                                          int(self.y+self.rect_size) )

                            return ExtractSubprocessor.Data (filepath, rects=[self.rect], landmarks_accurate=self.landmarks_accurate)

                else:
                    is_frame_done = True

                if is_frame_done:
                    self.result.append ( data )
                    self.input_data.pop(0)
                    io.progress_bar_inc(1)
                    self.extract_needed = True
                    self.rect_locked = False
        else:
            if len (self.input_data) > 0:
                return self.input_data.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        if not self.type != 'landmarks-manual':
            self.input_data.insert(0, data)

    def redraw(self):
        (h,w,c) = self.image.shape

        if not self.hide_help:
            image = cv2.addWeighted (self.image,1.0,self.text_lines_img,1.0,0)
        else:
            image = self.image.copy()

        view_rect = (np.array(self.rect) * self.view_scale).astype(np.int).tolist()
        view_landmarks  = (np.array(self.landmarks) * self.view_scale).astype(np.int).tolist()

        if self.rect_size <= 40:
            scaled_rect_size = h // 3 if w > h else w // 3

            p1 = (self.x - self.rect_size, self.y - self.rect_size)
            p2 = (self.x + self.rect_size, self.y - self.rect_size)
            p3 = (self.x - self.rect_size, self.y + self.rect_size)

            wh = h if h < w else w
            np1 = (w / 2 - wh / 4, h / 2 - wh / 4)
            np2 = (w / 2 + wh / 4, h / 2 - wh / 4)
            np3 = (w / 2 - wh / 4, h / 2 + wh / 4)

            mat = cv2.getAffineTransform( np.float32([p1,p2,p3])*self.view_scale, np.float32([np1,np2,np3]) )
            image = cv2.warpAffine(image, mat,(w,h) )
            view_landmarks = LandmarksProcessor.transform_points (view_landmarks, mat)

        landmarks_color = (255,255,0) if self.rect_locked else (0,255,0)
        LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, self.face_type, self.image_size, landmarks_color=landmarks_color)
        self.extract_needed = False

        io.show_image (self.wnd_name, image)


    #override
    def on_result (self, host_dict, data, result):
        if self.type == 'landmarks-manual':
            filepath, landmarks = result.filepath, result.landmarks

            if len(landmarks) != 0 and landmarks[0] is not None:
                self.landmarks = landmarks[0]

            self.redraw()
        else:
            self.result.append ( result )
            io.progress_bar_inc(1)



    #override
    def get_result(self):
        return self.result


class DeletedFilesSearcherSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.debug_paths_stems = client_dict['debug_paths_stems']
            return None

        #override
        def process_data(self, data):
            input_path_stem = Path(data[0]).stem
            return any ( [ input_path_stem == d_stem for d_stem in self.debug_paths_stems] )

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, input_paths, debug_paths ):
        self.input_paths = input_paths
        self.debug_paths_stems = [ Path(d).stem for d in debug_paths]
        self.result = []
        super().__init__('DeletedFilesSearcherSubprocessor', DeletedFilesSearcherSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8)):
            yield 'CPU%d' % (i), {}, {'debug_paths_stems' : self.debug_paths_stems}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Searching deleted files", len (self.input_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.input_paths) > 0:
            return [self.input_paths.pop(0)]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.input_paths.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        if result == False:
            self.result.append( data[0] )
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

def main(detector=None,
         input_path=None,
         output_path=None,
         output_debug=None,
         manual_fix=False,
         manual_output_debug_fix=False,
         manual_window_size=1368,
         face_type='full_face',
         max_faces_from_image=None,
         image_size=None,
         jpeg_quality=None,
         cpu_only = False,
         force_gpu_idxs = None,
         ):

    if not input_path.exists():
        io.log_err ('Input directory not found. Please ensure it exists.')
        return

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if face_type is not None:
        face_type = FaceType.fromString(face_type)

    if face_type is None:
        if manual_output_debug_fix:
            files = pathex.get_image_paths(output_path)
            if len(files) != 0:
                dflimg = DFLIMG.load(Path(files[0]))
                if dflimg is not None and dflimg.has_data():
                     face_type = FaceType.fromString ( dflimg.get_face_type() )

    input_image_paths = pathex.get_image_unique_filestem_paths(input_path, verbose_print_func=io.log_info)
    output_images_paths = pathex.get_image_paths(output_path)
    output_debug_path = output_path.parent / (output_path.name + '_debug')

    continue_extraction = False
    if not manual_output_debug_fix and len(output_images_paths) > 0:
        if len(output_images_paths) > 128:
            continue_extraction = io.input_bool ("Continue extraction?", True, help_message="Extraction can be continued, but you must specify the same options again.")

        if len(output_images_paths) > 128 and continue_extraction:
            try:
                input_image_paths = input_image_paths[ [ Path(x).stem for x in input_image_paths ].index ( Path(output_images_paths[-128]).stem.split('_')[0] ) : ]
            except:
                io.log_err("Error in fetching the last index. Extraction cannot be continued.")
                return
        elif input_path != output_path:
                io.input(f"\n WARNING !!! \n {output_path} contains files! \n They will be deleted. \n Press enter to continue.\n")
                for filename in output_images_paths:
                    Path(filename).unlink()

    device_config = nn.DeviceConfig.GPUIndexes( force_gpu_idxs or nn.ask_choose_device_idxs(choose_only_one=detector=='manual', suggest_all_gpu=True) ) \
                    if not cpu_only else nn.DeviceConfig.CPU()

    if face_type is None:
        face_type = io.input_str ("Face type", 'wf', ['f','wf','head'], help_message="Full face / whole face / head. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()
        face_type = {'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]

    if max_faces_from_image is None:
        max_faces_from_image = io.input_int(f"Max number of faces from image", 0, help_message="If you extract a src faceset that has frames with a large number of faces, it is advisable to set max faces to 3 to speed up extraction. 0 - unlimited")

    if image_size is None:
        image_size = io.input_int(f"Image size", 512 if face_type < FaceType.HEAD else 768, valid_range=[256,2048], help_message="Output image size. The higher image size, the worse face-enhancer works. Use higher than 512 value only if the source image is sharp enough and the face does not need to be enhanced.")

    if jpeg_quality is None:
        jpeg_quality = io.input_int(f"Jpeg quality", 90, valid_range=[1,100], help_message="Jpeg quality. The higher jpeg quality the larger the output file size.")

    if detector is None:
        io.log_info ("Choose detector type.")
        io.log_info ("[0] S3FD")
        io.log_info ("[1] manual")
        detector = {0:'s3fd', 1:'manual'}[ io.input_int("", 0, [0,1]) ]


    if output_debug is None:
        output_debug = io.input_bool (f"Write debug images to {output_debug_path.name}?", False)

    if output_debug:
        output_debug_path.mkdir(parents=True, exist_ok=True)

    if manual_output_debug_fix:
        if not output_debug_path.exists():
            io.log_err(f'{output_debug_path} not found. Re-extract faces with "Write debug images" option.')
            return
        else:
            detector = 'manual'
            io.log_info('Performing re-extract frames which were deleted from _debug directory.')

            input_image_paths = DeletedFilesSearcherSubprocessor (input_image_paths, pathex.get_image_paths(output_debug_path) ).run()
            input_image_paths = sorted (input_image_paths)
            io.log_info('Found %d images.' % (len(input_image_paths)))
    else:
        if not continue_extraction and output_debug_path.exists():
            for filename in pathex.get_image_paths(output_debug_path):
                Path(filename).unlink()

    images_found = len(input_image_paths)
    faces_detected = 0
    if images_found != 0:
        if detector == 'manual':
            io.log_info ('Performing manual extract...')
            data = ExtractSubprocessor ([ ExtractSubprocessor.Data(Path(filename)) for filename in input_image_paths ], 'landmarks-manual', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, manual_window_size=manual_window_size, device_config=device_config).run()

            io.log_info ('Performing 3rd pass...')
            data = ExtractSubprocessor (data, 'final', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, final_output_path=output_path, device_config=device_config).run()

        else:
            io.log_info ('Extracting faces...')
            data = ExtractSubprocessor ([ ExtractSubprocessor.Data(Path(filename)) for filename in input_image_paths ],
                                         'all',
                                         image_size,
                                         jpeg_quality,
                                         face_type,
                                         output_debug_path if output_debug else None,
                                         max_faces_from_image=max_faces_from_image,
                                         final_output_path=output_path,
                                         device_config=device_config).run()

        faces_detected += sum([d.faces_detected for d in data])

        if manual_fix:
            if all ( np.array ( [ d.faces_detected > 0 for d in data] ) == True ):
                io.log_info ('All faces are detected, manual fix not needed.')
            else:
                fix_data = [ ExtractSubprocessor.Data(d.filepath) for d in data if d.faces_detected == 0 ]
                io.log_info ('Performing manual fix for %d images...' % (len(fix_data)) )
                fix_data = ExtractSubprocessor (fix_data, 'landmarks-manual', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, manual_window_size=manual_window_size, device_config=device_config).run()
                fix_data = ExtractSubprocessor (fix_data, 'final', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, final_output_path=output_path, device_config=device_config).run()
                faces_detected += sum([d.faces_detected for d in fix_data])


    io.log_info ('-------------------------')
    io.log_info ('Images found:        %d' % (images_found) )
    io.log_info ('Faces detected:      %d' % (faces_detected) )
    io.log_info ('-------------------------')
