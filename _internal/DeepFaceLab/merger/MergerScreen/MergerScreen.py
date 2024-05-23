import math
from pathlib import Path

import numpy as np

from core import imagelib
from core.interact import interact as io
from core.cv2ex import *
from core import osex


class ScreenAssets(object):
    waiting_icon_image = cv2_imread ( str(Path(__file__).parent / 'gfx' / 'sand_clock_64.png') )

    @staticmethod
    def build_checkerboard_a( sh, size=5):
        h,w = sh[0], sh[1]
        tile = np.array([[0,1],[1,0]]).repeat(size, axis=0).repeat(size, axis=1)
        grid = np.tile(tile,(int(math.ceil((h+0.0)/(2*size))),int(math.ceil((w+0.0)/(2*size)))))
        return grid[:h,:w,None]

class Screen(object):
    def __init__(self, initial_scale_to_width=0, initial_scale_to_height=0, image=None, waiting_icon=False, **kwargs):
        self.initial_scale_to_width = initial_scale_to_width
        self.initial_scale_to_height = initial_scale_to_height
        self.image = image
        self.waiting_icon = waiting_icon

        self.state = -1
        self.scale = 1
        self.force_update = True
        self.is_first_appear = True
        self.show_checker_board = False

        self.last_screen_shape = (480,640,3)
        self.checkerboard_image = None
        self.set_image (image)
        self.scrn_manager = None

    def set_waiting_icon(self, b):
        self.waiting_icon = b

    def toggle_show_checker_board(self):
        self.show_checker_board = not self.show_checker_board
        self.force_update = True
    
    def get_image(self):
        return self.image
        
    def set_image(self, img):
        if not img is self.image:
            self.force_update = True

        self.image = img

        if self.image is not None:
            self.last_screen_shape = self.image.shape

            if self.initial_scale_to_width != 0:
                if self.last_screen_shape[1] > self.initial_scale_to_width:
                    self.scale = self.initial_scale_to_width / self.last_screen_shape[1]
                    self.force_update = True
                self.initial_scale_to_width = 0

            elif self.initial_scale_to_height != 0:
                if self.last_screen_shape[0] > self.initial_scale_to_height:
                    self.scale = self.initial_scale_to_height / self.last_screen_shape[0]
                    self.force_update = True
                self.initial_scale_to_height = 0


    def diff_scale(self, diff):
        self.scale = np.clip (self.scale + diff, 0.1, 4.0)
        self.force_update = True

    def show(self, force=False):
        new_state = 0 | self.waiting_icon

        if self.state != new_state or self.force_update or force:
            self.state = new_state
            self.force_update = False

            if self.image is None:
                screen = np.zeros ( self.last_screen_shape, dtype=np.uint8 )
            else:
                screen = self.image.copy()

            if self.waiting_icon:
                imagelib.overlay_alpha_image (screen, ScreenAssets.waiting_icon_image, (0,0) )

            h,w,c = screen.shape
            if self.scale != 1.0:
                screen = cv2.resize ( screen, ( int(w*self.scale), int(h*self.scale) ) )

            if c == 4:
                if not self.show_checker_board:
                    screen = screen[...,0:3]
                else:
                    if self.checkerboard_image is None or self.checkerboard_image.shape[0:2] != screen.shape[0:2]:
                        self.checkerboard_image = ScreenAssets.build_checkerboard_a(screen.shape)

                    screen = screen[...,0:3]*0.75 + 64*self.checkerboard_image*(1- (screen[...,3:4].astype(np.float32)/255.0) )
                    screen = screen.astype(np.uint8)

            io.show_image(self.scrn_manager.wnd_name, screen)

            if self.is_first_appear:
                self.is_first_appear = False
                #center window
                desktop_w, desktop_h = osex.get_screen_size()
                h,w,c = screen.shape
                cv2.moveWindow(self.scrn_manager.wnd_name, max(0,(desktop_w-w) // 2), max(0, (desktop_h-h) // 2) )

            io.process_messages(0.0001)

class ScreenManager(object):
    def __init__(self, window_name="ScreenManager", screens=None, capture_keys=False ):
        self.screens = screens or []
        self.current_screen_id = 0

        if self.screens is not None:
            for screen in self.screens:
                screen.scrn_manager = self

        self.wnd_name = window_name
        io.named_window(self.wnd_name)


        if capture_keys:
            io.capture_keys(self.wnd_name)

    def finalize(self):
        io.destroy_all_windows()

    def get_key_events(self):
        return io.get_key_events(self.wnd_name)

    def switch_screens(self):
        self.current_screen_id = (self.current_screen_id + 1) % len(self.screens)
        self.screens[self.current_screen_id].show(force=True)

    def show_current(self):
        self.screens[self.current_screen_id].show()

    def get_current(self):
        return self.screens[self.current_screen_id]

    def set_current(self, screen):
        self.current_screen_id = self.screens.index(screen)
