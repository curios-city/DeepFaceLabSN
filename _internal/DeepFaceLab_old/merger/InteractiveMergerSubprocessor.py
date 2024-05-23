import multiprocessing
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from merger import MergeFaceAvatar, MergeMasked, MergerConfig

from .MergerScreen import Screen, ScreenManager

MERGER_DEBUG = False
class InteractiveMergerSubprocessor(Subprocessor):
    #print("MERGER_DEBUG = False")
    class Frame(object):
        def __init__(self, prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None):
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = None
            self.output_mask_filepath = None

            self.idx = None
            self.cfg = None
            self.is_done = False
            self.is_processing = False
            self.is_shown = False
            self.image = None

    class ProcessingFrame(object):
        def __init__(self, idx=None,
                           cfg=None,
                           prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None,
                           output_filepath=None,
                           output_mask_filepath=None,
                           need_return_image = False):
            self.idx = idx
            self.cfg = cfg
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = output_filepath
            self.output_mask_filepath = output_mask_filepath

            self.need_return_image = need_return_image
            if self.need_return_image:
                self.image = None

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            #self.log_info ('运行在 %s.' % (client_dict['device_name']) )
            self.device_idx  = client_dict['device_idx']
            self.device_name = client_dict['device_name']
            self.predictor_func = client_dict['predictor_func']
            self.predictor_input_shape = client_dict['predictor_input_shape']
            self.face_enhancer_func = client_dict['face_enhancer_func']
            self.xseg_256_extract_func = client_dict['xseg_256_extract_func']


            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None:
                sys.stdin = os.fdopen(stdin_fd)

            return None

        #override
        def process_data(self, pf): #pf=ProcessingFrame
            cfg = pf.cfg.copy()

            frame_info = pf.frame_info
            filepath = frame_info.filepath

            if len(frame_info.landmarks_list) == 0:
                
                if cfg.mode == 'raw-predict':        
                    h,w,c = self.predictor_input_shape
                    img_bgr = np.zeros( (h,w,3), dtype=np.uint8)
                    img_mask = np.zeros( (h,w,1), dtype=np.uint8)               
                else:
                    self.log_info (f' {filepath.name} 提取面部时没有提取到脸或者脸已经被删除，合成时这一帧没有面孔，请检查该帧有无需要更换的人脸（此提示不是报错）')
                    img_bgr = cv2_imread(filepath)
                    imagelib.normalize_channels(img_bgr, 3)                    
                    h,w,c = img_bgr.shape
                    img_mask = np.zeros( (h,w,1), dtype=img_bgr.dtype)
                    
                cv2_imwrite (pf.output_filepath, img_bgr)
                cv2_imwrite (pf.output_mask_filepath, img_mask)

                if pf.need_return_image:
                    pf.image = np.concatenate ([img_bgr, img_mask], axis=-1)

            else:
                if cfg.type == MergerConfig.TYPE_MASKED:
                    try:
                        final_img = MergeMasked (self.predictor_func, self.predictor_input_shape,
                                                 face_enhancer_func=self.face_enhancer_func,
                                                 xseg_256_extract_func=self.xseg_256_extract_func,
                                                 cfg=cfg,
                                                 frame_info=frame_info)
                    except Exception as e:
                        e_str = traceback.format_exc()
                        if 'MemoryError' in e_str:
                            raise Subprocessor.SilenceException
                        else:
                            raise Exception( f'合并文件时出错 [{filepath}]: {e_str}' )

                elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
                    final_img = MergeFaceAvatar (self.predictor_func, self.predictor_input_shape,
                                                   cfg, pf.prev_temporal_frame_infos,
                                                        pf.frame_info,
                                                        pf.next_temporal_frame_infos )

                cv2_imwrite (pf.output_filepath,      final_img[...,0:3] )
                cv2_imwrite (pf.output_mask_filepath, final_img[...,3:4] )

                if pf.need_return_image:
                    pf.image = final_img

            return pf

        #overridable
        def get_data_name (self, pf):
            #return string identificator of your data
            return pf.frame_info.filepath




    #override
    def __init__(self, is_interactive, merger_session_filepath, predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, merger_config, frames, frames_root_path, output_path, output_mask_path, model_iter, subprocess_count=4):
        if len (frames) == 0:
            raise ValueError ("len (frames) == 0")

        super().__init__('Merger', InteractiveMergerSubprocessor.Cli, io_loop_sleep_time=0.001)

        self.is_interactive = is_interactive
        self.merger_session_filepath = Path(merger_session_filepath)
        self.merger_config = merger_config

        self.predictor_func = predictor_func
        self.predictor_input_shape = predictor_input_shape

        self.face_enhancer_func = face_enhancer_func
        self.xseg_256_extract_func = xseg_256_extract_func

        self.frames_root_path = frames_root_path
        self.output_path = output_path
        self.output_mask_path = output_mask_path
        self.model_iter = model_iter

        self.prefetch_frame_count = self.process_count = subprocess_count

        session_data = None
        if self.is_interactive and self.merger_session_filepath.exists():
            io.input_skip_pending()
            if io.input_bool ("使用保存的会话?", True):
                try:
                    with open( str(self.merger_session_filepath), "rb") as f:
                        session_data = pickle.loads(f.read())

                except Exception as e:
                    pass

        rewind_to_frame_idx = None
        self.frames = frames
        self.frames_idxs = [ *range(len(self.frames)) ]
        self.frames_done_idxs = []

        if self.is_interactive and session_data is not None:
            # Loaded session data, check it
            s_frames = session_data.get('frames', None)
            s_frames_idxs = session_data.get('frames_idxs', None)
            s_frames_done_idxs = session_data.get('frames_done_idxs', None)
            s_model_iter = session_data.get('model_iter', None)

            frames_equal = (s_frames is not None) and \
                           (s_frames_idxs is not None) and \
                           (s_frames_done_idxs is not None) and \
                           (s_model_iter is not None) and \
                           (len(frames) == len(s_frames)) # frames count must match

            if frames_equal:
                for i in range(len(frames)):
                    frame = frames[i]
                    s_frame = s_frames[i]
                    # frames filenames must match
                    if frame.frame_info.filepath.name != s_frame.frame_info.filepath.name:
                        frames_equal = False
                    if not frames_equal:
                        break

            if frames_equal:
                io.log_info ('U使用保存的会话来自 ' + '/'.join (self.merger_session_filepath.parts[-2:]) )

                for frame in s_frames:
                    if frame.cfg is not None:
                        # recreate MergerConfig class using constructor with get_config() as dict params
                        # so if any new param will be added, old merger session will work properly
                        frame.cfg = frame.cfg.__class__( **frame.cfg.get_config() )

                self.frames = s_frames
                self.frames_idxs = s_frames_idxs
                self.frames_done_idxs = s_frames_done_idxs

                if self.model_iter != s_model_iter:
                    # model was more trained, recompute all frames
                    rewind_to_frame_idx = -1
                    for frame in self.frames:
                        frame.is_done = False
                elif len(self.frames_idxs) == 0:
                    # all frames are done?
                    rewind_to_frame_idx = -1

                if len(self.frames_idxs) != 0:
                    cur_frame = self.frames[self.frames_idxs[0]]
                    cur_frame.is_shown = False

            if not frames_equal:
                session_data = None

        if session_data is None:
            for filename in pathex.get_image_paths(self.output_path): #remove all images in output_path
                Path(filename).unlink()

            for filename in pathex.get_image_paths(self.output_mask_path): #remove all images in output_mask_path
                Path(filename).unlink()


            frames[0].cfg = self.merger_config.copy()

        for i in range( len(self.frames) ):
            frame = self.frames[i]
            frame.idx = i
            frame.output_filepath      = self.output_path      / ( frame.frame_info.filepath.stem + '.png' )
            frame.output_mask_filepath = self.output_mask_path / ( frame.frame_info.filepath.stem + '.png' )

            if not frame.output_filepath.exists() or \
               not frame.output_mask_filepath.exists():
                # if some frame does not exist, recompute and rewind
                frame.is_done = False
                frame.is_shown = False

                if rewind_to_frame_idx is None:
                    rewind_to_frame_idx = i-1
                else:
                    rewind_to_frame_idx = min(rewind_to_frame_idx, i-1)

        if rewind_to_frame_idx is not None:
            while len(self.frames_done_idxs) > 0:
                if self.frames_done_idxs[-1] > rewind_to_frame_idx:
                    prev_frame = self.frames[self.frames_done_idxs.pop()]
                    self.frames_idxs.insert(0, prev_frame.idx)
                else:
                    break
    #override
    def process_info_generator(self):
        r = [0] if MERGER_DEBUG else range(self.process_count)

        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'predictor_func': self.predictor_func,
                                      'predictor_input_shape' : self.predictor_input_shape,
                                      'face_enhancer_func': self.face_enhancer_func,
                                      'xseg_256_extract_func' : self.xseg_256_extract_func,
                                      'stdin_fd': sys.stdin.fileno() if MERGER_DEBUG else None
                                      }

    #overridable optional
    def on_clients_initialized(self):
        io.progress_bar ("合成进度", len(self.frames_idxs)+len(self.frames_done_idxs), initial=len(self.frames_done_idxs) )

        self.process_remain_frames = not self.is_interactive
        self.is_interactive_quitting = not self.is_interactive

        if self.is_interactive:
            help_images = {
                    MergerConfig.TYPE_MASKED :      cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_masked.jpg') ),
                    MergerConfig.TYPE_FACE_AVATAR : cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_face_avatar.jpg') ),
                }

            self.main_screen = Screen(initial_scale_to_width=1368, image=None, waiting_icon=True)
            self.help_screen = Screen(initial_scale_to_height=768, image=help_images[self.merger_config.type], waiting_icon=False)
            self.screen_manager = ScreenManager( "Merger", [self.main_screen, self.help_screen], capture_keys=True )
            self.screen_manager.set_current (self.help_screen)
            self.screen_manager.show_current()

            self.masked_keys_funcs = {
                    '`' : lambda cfg,shift_pressed: cfg.set_mode(0),
                    '1' : lambda cfg,shift_pressed: cfg.set_mode(1),
                    '2' : lambda cfg,shift_pressed: cfg.set_mode(2),
                    '3' : lambda cfg,shift_pressed: cfg.set_mode(3),
                    '4' : lambda cfg,shift_pressed: cfg.set_mode(4),
                    '5' : lambda cfg,shift_pressed: cfg.set_mode(5),
                    '6' : lambda cfg,shift_pressed: cfg.set_mode(6),
                    'q' : lambda cfg,shift_pressed: cfg.add_hist_match_threshold(1 if not shift_pressed else 5),
                    'a' : lambda cfg,shift_pressed: cfg.add_hist_match_threshold(-1 if not shift_pressed else -5),
                    'w' : lambda cfg,shift_pressed: cfg.add_erode_mask_modifier(1 if not shift_pressed else 5),
                    's' : lambda cfg,shift_pressed: cfg.add_erode_mask_modifier(-1 if not shift_pressed else -5),
                    'e' : lambda cfg,shift_pressed: cfg.add_blur_mask_modifier(1 if not shift_pressed else 5),
                    'd' : lambda cfg,shift_pressed: cfg.add_blur_mask_modifier(-1 if not shift_pressed else -5),
                    'r' : lambda cfg,shift_pressed: cfg.add_motion_blur_power(1 if not shift_pressed else 5),
                    'f' : lambda cfg,shift_pressed: cfg.add_motion_blur_power(-1 if not shift_pressed else -5),
                    't' : lambda cfg,shift_pressed: cfg.add_super_resolution_power(1 if not shift_pressed else 5),
                    'g' : lambda cfg,shift_pressed: cfg.add_super_resolution_power(-1 if not shift_pressed else -5),
                    'y' : lambda cfg,shift_pressed: cfg.add_blursharpen_amount(1 if not shift_pressed else 5),
                    'h' : lambda cfg,shift_pressed: cfg.add_blursharpen_amount(-1 if not shift_pressed else -5),
                    'u' : lambda cfg,shift_pressed: cfg.add_output_face_scale(1 if not shift_pressed else 5),
                    'j' : lambda cfg,shift_pressed: cfg.add_output_face_scale(-1 if not shift_pressed else -5),
                    'i' : lambda cfg,shift_pressed: cfg.add_image_denoise_power(1 if not shift_pressed else 5),
                    'k' : lambda cfg,shift_pressed: cfg.add_image_denoise_power(-1 if not shift_pressed else -5),
                    'o' : lambda cfg,shift_pressed: cfg.add_bicubic_degrade_power(1 if not shift_pressed else 5),
                    'l' : lambda cfg,shift_pressed: cfg.add_bicubic_degrade_power(-1 if not shift_pressed else -5),
                    'p' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(1 if not shift_pressed else 5),
                    ';' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(-1),
                    ':' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(-5),
                    'z' : lambda cfg,shift_pressed: cfg.toggle_masked_hist_match(),
                    'x' : lambda cfg,shift_pressed: cfg.toggle_mask_mode(),
                    'c' : lambda cfg,shift_pressed: cfg.toggle_color_transfer_mode(),
                    'n' : lambda cfg,shift_pressed: cfg.toggle_sharpen_mode(),
                    }
            self.masked_keys = list(self.masked_keys_funcs.keys())

    #overridable optional
    def on_clients_finalized(self):
        io.progress_bar_close()

        if self.is_interactive:
            self.screen_manager.finalize()

            for frame in self.frames:
                frame.output_filepath = None
                frame.output_mask_filepath = None
                frame.image = None

            session_data = {
                'frames': self.frames,
                'frames_idxs': self.frames_idxs,
                'frames_done_idxs': self.frames_done_idxs,
                'model_iter' : self.model_iter,
            }
            self.merger_session_filepath.write_bytes( pickle.dumps(session_data) )

            io.log_info ("会话保存到 " + '/'.join (self.merger_session_filepath.parts[-2:]) )

    #override
    def on_tick(self):
        io.process_messages()

        go_prev_frame = False
        go_first_frame = False
        go_prev_frame_overriding_cfg = False
        go_first_frame_overriding_cfg = False

        go_next_frame = self.process_remain_frames
        go_next_frame_overriding_cfg = False
        go_last_frame_overriding_cfg = False

        cur_frame = None
        if len(self.frames_idxs) != 0:
            cur_frame = self.frames[self.frames_idxs[0]]

        if self.is_interactive:

            screen_image = None if self.process_remain_frames else \
                                   self.main_screen.get_image()

            self.main_screen.set_waiting_icon( self.process_remain_frames or \
                                               self.is_interactive_quitting )

            if cur_frame is not None and not self.is_interactive_quitting:

                if not self.process_remain_frames:
                    if cur_frame.is_done:
                        if not cur_frame.is_shown:
                            if cur_frame.image is None:
                                image      = cv2_imread (cur_frame.output_filepath, verbose=False)
                                image_mask = cv2_imread (cur_frame.output_mask_filepath, verbose=False)
                                if image is None or image_mask is None:
                                    # unable to read? recompute then
                                    cur_frame.is_done = False
                                else:
                                    image = imagelib.normalize_channels(image, 3)
                                    image_mask = imagelib.normalize_channels(image_mask, 1)
                                    cur_frame.image = np.concatenate([image, image_mask], -1)

                            if cur_frame.is_done:
                                io.log_info (cur_frame.cfg.to_string( cur_frame.frame_info.filepath.name) )
                                cur_frame.is_shown = True
                                screen_image = cur_frame.image
                    else:
                        self.main_screen.set_waiting_icon(True)

            self.main_screen.set_image(screen_image)
            self.screen_manager.show_current()

            key_events = self.screen_manager.get_key_events()
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == 9: #tab
                self.screen_manager.switch_screens()
            else:
                if key == 27: #esc
                    self.is_interactive_quitting = True
                elif self.screen_manager.get_current() is self.main_screen:

                    if self.merger_config.type == MergerConfig.TYPE_MASKED and chr_key in self.masked_keys:
                        self.process_remain_frames = False

                        if cur_frame is not None:
                            cfg = cur_frame.cfg
                            prev_cfg = cfg.copy()

                            if cfg.type == MergerConfig.TYPE_MASKED:
                                self.masked_keys_funcs[chr_key](cfg, shift_pressed)

                            if prev_cfg != cfg:
                                io.log_info ( cfg.to_string(cur_frame.frame_info.filepath.name) )
                                cur_frame.is_done = False
                                cur_frame.is_shown = False
                    else:

                        if chr_key == ',' or chr_key == 'm':
                            self.process_remain_frames = False
                            go_prev_frame = True

                            if chr_key == ',':
                                if shift_pressed:
                                    go_first_frame = True

                            elif chr_key == 'm':
                                if not shift_pressed:
                                    go_prev_frame_overriding_cfg = True
                                else:
                                    go_first_frame_overriding_cfg = True

                        elif chr_key == '.' or chr_key == '/':
                            self.process_remain_frames = False
                            go_next_frame = True

                            if chr_key == '.':
                                if shift_pressed:
                                    self.process_remain_frames = not self.process_remain_frames

                            elif chr_key == '/':
                                if not shift_pressed:
                                    go_next_frame_overriding_cfg = True
                                else:
                                    go_last_frame_overriding_cfg = True

                        elif chr_key == '-':
                            self.screen_manager.get_current().diff_scale(-0.1)
                        elif chr_key == '=':
                            self.screen_manager.get_current().diff_scale(0.1)
                        elif chr_key == 'v':
                            self.screen_manager.get_current().toggle_show_checker_board()

        if go_prev_frame:
            if cur_frame is None or cur_frame.is_done:
                if cur_frame is not None:
                    cur_frame.image = None

                while True:
                    if len(self.frames_done_idxs) > 0:
                        prev_frame = self.frames[self.frames_done_idxs.pop()]
                        self.frames_idxs.insert(0, prev_frame.idx)
                        prev_frame.is_shown = False
                        io.progress_bar_inc(-1)

                        if cur_frame is not None and (go_prev_frame_overriding_cfg or go_first_frame_overriding_cfg):
                            if prev_frame.cfg != cur_frame.cfg:
                                prev_frame.cfg = cur_frame.cfg.copy()
                                prev_frame.is_done = False

                        cur_frame = prev_frame

                    if go_first_frame_overriding_cfg or go_first_frame:
                        if len(self.frames_done_idxs) > 0:
                            continue
                    break

        elif go_next_frame:
            if cur_frame is not None and cur_frame.is_done:
                cur_frame.image = None
                cur_frame.is_shown = True
                self.frames_done_idxs.append(cur_frame.idx)
                self.frames_idxs.pop(0)
                io.progress_bar_inc(1)

                f = self.frames

                if len(self.frames_idxs) != 0:
                    next_frame = f[ self.frames_idxs[0] ]
                    next_frame.is_shown = False

                    if go_next_frame_overriding_cfg or go_last_frame_overriding_cfg:

                        if go_next_frame_overriding_cfg:
                            to_frames = next_frame.idx+1
                        else:
                            to_frames = len(f)

                        for i in range( next_frame.idx, to_frames ):
                            f[i].cfg = None

                    for i in range( min(len(self.frames_idxs), self.prefetch_frame_count) ):
                        frame = f[ self.frames_idxs[i] ]
                        if frame.cfg is None:
                            if i == 0:
                                frame.cfg = cur_frame.cfg.copy()
                            else:
                                frame.cfg = f[ self.frames_idxs[i-1] ].cfg.copy()

                            frame.is_done = False #initiate solve again
                            frame.is_shown = False

            if len(self.frames_idxs) == 0:
                self.process_remain_frames = False

        return (self.is_interactive and self.is_interactive_quitting) or \
               (not self.is_interactive and self.process_remain_frames == False)


    #override
    def on_data_return (self, host_dict, pf):
        frame = self.frames[pf.idx]
        frame.is_done = False
        frame.is_processing = False

    #override
    def on_result (self, host_dict, pf_sent, pf_result):
        frame = self.frames[pf_result.idx]
        frame.is_processing = False
        if frame.cfg == pf_result.cfg:
            frame.is_done = True
            frame.image = pf_result.image

    #override
    def get_data(self, host_dict):
        if self.is_interactive and self.is_interactive_quitting:
            return None

        for i in range ( min(len(self.frames_idxs), self.prefetch_frame_count) ):
            frame = self.frames[ self.frames_idxs[i] ]

            if not frame.is_done and not frame.is_processing and frame.cfg is not None:
                frame.is_processing = True
                return InteractiveMergerSubprocessor.ProcessingFrame(idx=frame.idx,
                                                           cfg=frame.cfg.copy(),
                                                           prev_temporal_frame_infos=frame.prev_temporal_frame_infos,
                                                           frame_info=frame.frame_info,
                                                           next_temporal_frame_infos=frame.next_temporal_frame_infos,
                                                           output_filepath=frame.output_filepath,
                                                           output_mask_filepath=frame.output_mask_filepath,
                                                           need_return_image=True )

        return None

    #override
    def get_result(self):
        return 0