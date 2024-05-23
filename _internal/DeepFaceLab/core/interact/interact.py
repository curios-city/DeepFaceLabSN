import multiprocessing
import os
import sys
import threading
import time
import types

import colorama
import cv2
import numpy as np
from tqdm import tqdm

from core import stdex

try:
    import IPython #if success we are in colab
    from IPython.display import display, clear_output
    import PIL
    import matplotlib.pyplot as plt
    is_colab = True
except:
    is_colab = False

yn_str = {True:'y',False:'n'}

class InteractBase(object):
    EVENT_LBUTTONDOWN = 1
    EVENT_LBUTTONUP = 2
    EVENT_MBUTTONDOWN = 3
    EVENT_MBUTTONUP = 4
    EVENT_RBUTTONDOWN = 5
    EVENT_RBUTTONUP = 6
    EVENT_MOUSEWHEEL = 10

    def __init__(self):
        self.named_windows = {}
        self.capture_mouse_windows = {}
        self.capture_keys_windows = {}
        self.mouse_events = {}
        self.key_events = {}
        self.pg_bar = None
        self.focus_wnd_name = None
        self.error_log_line_prefix = '/!\\ '

        self.process_messages_callbacks = {}

    def is_support_windows(self):
        return False

    def is_colab(self):
        return False

    def on_destroy_all_windows(self):
        raise NotImplemented

    def on_create_window (self, wnd_name):
        raise NotImplemented

    def on_destroy_window (self, wnd_name):
        raise NotImplemented

    def on_show_image (self, wnd_name, img):
        raise NotImplemented

    def on_capture_mouse (self, wnd_name):
        raise NotImplemented

    def on_capture_keys (self, wnd_name):
        raise NotImplemented

    def on_process_messages(self, sleep_time=0):
        raise NotImplemented

    def on_wait_any_key(self):
        raise NotImplemented

    def log_info(self, msg, end='\n'):
        if self.pg_bar is not None:
            print ("\n")
        print (msg, end=end)

    def log_err(self, msg, end='\n'):
        if self.pg_bar is not None:
            print ("\n")
        print (f'{self.error_log_line_prefix}{msg}', end=end)

    def named_window(self, wnd_name):
        if wnd_name not in self.named_windows:
            #we will show window only on first show_image
            self.named_windows[wnd_name] = 0
            self.focus_wnd_name = wnd_name
        else: print("named_window: ", wnd_name, " already created.")

    def destroy_all_windows(self):
        if len( self.named_windows ) != 0:
            self.on_destroy_all_windows()
            self.named_windows = {}
            self.capture_mouse_windows = {}
            self.capture_keys_windows = {}
            self.mouse_events = {}
            self.key_events = {}
            self.focus_wnd_name = None

    def destroy_window(self, wnd_name):
        if wnd_name in self.named_windows:
            self.on_destroy_window(wnd_name)
            self.named_windows.pop(wnd_name)

            if wnd_name == self.focus_wnd_name:
                self.focus_wnd_name = list(self.named_windows.keys())[-1] if len( self.named_windows ) != 0 else None

            if wnd_name in self.capture_mouse_windows:
                self.capture_mouse_windows.pop(wnd_name)

            if wnd_name in self.capture_keys_windows:
                self.capture_keys_windows.pop(wnd_name)

            if wnd_name in self.mouse_events:
                self.mouse_events.pop(wnd_name)

            if wnd_name in self.key_events:
                self.key_events.pop(wnd_name)

    def show_image(self, wnd_name, img):
        if wnd_name in self.named_windows:
            if self.named_windows[wnd_name] == 0:
                self.named_windows[wnd_name] = 1
                self.on_create_window(wnd_name)
                if wnd_name in self.capture_mouse_windows:
                    self.capture_mouse(wnd_name)
            self.on_show_image(wnd_name,img)
        else: print("show_image: named_window ", wnd_name, " not found.")

    def capture_mouse(self, wnd_name):
        if wnd_name in self.named_windows:
            self.capture_mouse_windows[wnd_name] = True
            if self.named_windows[wnd_name] == 1:
                self.on_capture_mouse(wnd_name)
        else: print("capture_mouse: named_window ", wnd_name, " not found.")

    def capture_keys(self, wnd_name):
        if wnd_name in self.named_windows:
            if wnd_name not in self.capture_keys_windows:
                self.capture_keys_windows[wnd_name] = True
                self.on_capture_keys(wnd_name)
            else: print("capture_keys: already set for window ", wnd_name)
        else: print("capture_keys: named_window ", wnd_name, " not found.")

    def progress_bar(self, desc, total, leave=True, initial=0):
        if self.pg_bar is None:
            self.pg_bar = tqdm( total=total, desc=desc, leave=leave, ascii=True, initial=initial )
        else: print("progress_bar: already set.")

    def progress_bar_inc(self, c):
        if self.pg_bar is not None:
            self.pg_bar.n += c
            self.pg_bar.refresh()
        else: print("progress_bar not set.")

    def progress_bar_close(self):
        if self.pg_bar is not None:
            self.pg_bar.close()
            self.pg_bar = None
        else: print("progress_bar not set.")

    def progress_bar_generator(self, data, desc=None, leave=True, initial=0):
        self.pg_bar = tqdm( data, desc=desc, leave=leave, ascii=True, initial=initial )
        for x in self.pg_bar:
            yield x
        self.pg_bar.close()
        self.pg_bar = None

    def add_process_messages_callback(self, func ):
        tid = threading.get_ident()
        callbacks = self.process_messages_callbacks.get(tid, None)
        if callbacks is None:
            callbacks = []
            self.process_messages_callbacks[tid] = callbacks

        callbacks.append ( func )

    def process_messages(self, sleep_time=0):
        callbacks = self.process_messages_callbacks.get(threading.get_ident(), None)
        if callbacks is not None:
            for func in callbacks:
                func()

        self.on_process_messages(sleep_time)

    def wait_any_key(self):
        self.on_wait_any_key()

    def add_mouse_event(self, wnd_name, x, y, ev, flags):
        if wnd_name not in self.mouse_events:
            self.mouse_events[wnd_name] = []
        self.mouse_events[wnd_name] += [ (x, y, ev, flags) ]

    def add_key_event(self, wnd_name, ord_key, ctrl_pressed, alt_pressed, shift_pressed):
        if wnd_name not in self.key_events:
            self.key_events[wnd_name] = []
        self.key_events[wnd_name] += [ (ord_key, chr(ord_key) if ord_key <= 255 else chr(0), ctrl_pressed, alt_pressed, shift_pressed) ]

    def get_mouse_events(self, wnd_name):
        ar = self.mouse_events.get(wnd_name, [])
        self.mouse_events[wnd_name] = []
        return ar

    def get_key_events(self, wnd_name):
        ar = self.key_events.get(wnd_name, [])
        self.key_events[wnd_name] = []
        return ar

    def input(self, s):
        return input(s)

    def input_number(self, s, default_value, valid_list=None, show_default_value=True, add_info=None, help_message=None):
        if show_default_value and default_value is not None:
            s = f"[{default_value}] {s}"

        if add_info is not None or \
           help_message is not None:
            s += " ("

        if add_info is not None:
            s += f" {add_info}"
        if help_message is not None:
            s += " ?:help"

        if add_info is not None or \
           help_message is not None:
            s += " )"

        s += " : "

        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    result = default_value
                    break

                if help_message is not None and inp == '?':
                    print (help_message)
                    continue

                i = float(inp)
                if (valid_list is not None) and (i not in valid_list):
                    result = default_value
                    break
                result = i
                break
            except:
                result = default_value
                break

        print(result)
        return result

    def input_int(self, s, default_value, valid_range=None, valid_list=None, add_info=None, show_default_value=True, help_message=None):
        if show_default_value:
            if len(s) != 0:
                s = f"[{default_value}] {s}"
            else:
                s = f"[{default_value}]"

        if add_info is not None or \
           valid_range is not None or \
           help_message is not None:
            s += " ("

        if valid_range is not None:
            s += f" {valid_range[0]}-{valid_range[1]}"

        if add_info is not None:
            s += f" {add_info}"

        if help_message is not None:
            s += " ?:help"

        if add_info is not None or \
           valid_range is not None or \
           help_message is not None:
            s += " )"

        s += " : "

        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    raise ValueError("")

                if help_message is not None and inp == '?':
                    print (help_message)
                    continue

                i = int(inp)
                if valid_range is not None:
                    i = int(np.clip(i, valid_range[0], valid_range[1]))

                if (valid_list is not None) and (i not in valid_list):
                    i = default_value

                result = i
                break
            except:
                result = default_value
                break
        print (result)
        return result

    def input_bool(self, s, default_value, help_message=None):
        s = f"[{yn_str[default_value]}] {s} ( y/n"

        if help_message is not None:
            s += " ?:help"
        s += " ) : "

        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    raise ValueError("")

                if help_message is not None and inp == '?':
                    print (help_message)
                    continue

                return bool ( {"y":True,"n":False}.get(inp.lower(), default_value) )
            except:
                print ( "y" if default_value else "n" )
                return default_value

    def input_str(self, s, default_value=None, valid_list=None, show_default_value=True, help_message=None):
        if show_default_value and default_value is not None:
            s = f"[{default_value}] {s}"

        if valid_list is not None or \
           help_message is not None:
            s += " ("

        if valid_list is not None:
            s += " " + "/".join(valid_list)

        if help_message is not None:
            s += " ?:help"

        if valid_list is not None or \
           help_message is not None:
            s += " )"

        s += " : "


        while True:
            try:
                inp = input(s)

                if len(inp) == 0:
                    if default_value is None:
                        print("")
                        return None
                    result = default_value
                    break

                if help_message is not None and inp == '?':
                    print(help_message)
                    continue

                if valid_list is not None:
                    if inp.lower() in valid_list:
                        result = inp.lower()
                        break
                    if inp in valid_list:
                        result = inp
                        break
                    continue

                result = inp
                break
            except:
                result = default_value
                break

        print(result)
        return result

    def input_process(self, stdin_fd, sq, str):
        sys.stdin = os.fdopen(stdin_fd)
        try:
            inp = input (str)
            sq.put (True)
        except:
            sq.put (False)

    def input_in_time (self, str, max_time_sec):
        sq = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.input_process, args=( sys.stdin.fileno(), sq, str))
        p.daemon = True
        p.start()
        t = time.time()
        inp = False
        while True:
            if not sq.empty():
                inp = sq.get()
                break
            if time.time() - t > max_time_sec:
                break


        p.terminate()
        p.join()

        old_stdin = sys.stdin
        sys.stdin = os.fdopen( os.dup(sys.stdin.fileno()) )
        old_stdin.close()
        return inp

    def input_process_skip_pending(self, stdin_fd):
        sys.stdin = os.fdopen(stdin_fd)
        while True:
            try:
                if sys.stdin.isatty():
                    sys.stdin.read()
            except:
                pass

    def input_skip_pending(self):
        if is_colab:
            # currently it does not work on Colab
            return
        """
        skips unnecessary inputs between the dialogs
        """
        p = multiprocessing.Process(target=self.input_process_skip_pending, args=( sys.stdin.fileno(), ))
        p.daemon = True
        p.start()
        time.sleep(0.5)
        p.terminate()
        p.join()
        sys.stdin = os.fdopen( sys.stdin.fileno() )


class InteractDesktop(InteractBase):
    def __init__(self):
        colorama.init()
        super().__init__()

    def color_red(self):
        pass


    def is_support_windows(self):
        return True

    def on_destroy_all_windows(self):
        cv2.destroyAllWindows()

    def on_create_window (self, wnd_name):
        cv2.namedWindow(wnd_name)

    def on_destroy_window (self, wnd_name):
        cv2.destroyWindow(wnd_name)

    def on_show_image (self, wnd_name, img):
        cv2.imshow (wnd_name, img)

    def on_capture_mouse (self, wnd_name):
        self.last_xy = (0,0)

        def onMouse(event, x, y, flags, param):
            (inst, wnd_name) = param
            if event == cv2.EVENT_LBUTTONDOWN: ev = InteractBase.EVENT_LBUTTONDOWN
            elif event == cv2.EVENT_LBUTTONUP: ev = InteractBase.EVENT_LBUTTONUP
            elif event == cv2.EVENT_RBUTTONDOWN: ev = InteractBase.EVENT_RBUTTONDOWN
            elif event == cv2.EVENT_RBUTTONUP: ev = InteractBase.EVENT_RBUTTONUP
            elif event == cv2.EVENT_MBUTTONDOWN: ev = InteractBase.EVENT_MBUTTONDOWN
            elif event == cv2.EVENT_MBUTTONUP: ev = InteractBase.EVENT_MBUTTONUP
            elif event == cv2.EVENT_MOUSEWHEEL:
                ev = InteractBase.EVENT_MOUSEWHEEL
                x,y = self.last_xy #fix opencv bug when window size more than screen size
            else: ev = 0

            self.last_xy = (x,y)
            inst.add_mouse_event (wnd_name, x, y, ev, flags)
        cv2.setMouseCallback(wnd_name, onMouse, (self,wnd_name) )

    def on_capture_keys (self, wnd_name):
        pass

    def on_process_messages(self, sleep_time=0):

        has_windows = False
        has_capture_keys = False

        if len(self.named_windows) != 0:
            has_windows = True

        if len(self.capture_keys_windows) != 0:
            has_capture_keys = True

        if has_windows or has_capture_keys:
            wait_key_time = max(1, int(sleep_time*1000) )
            ord_key = cv2.waitKeyEx(wait_key_time)
            
            shift_pressed = False
            if ord_key != -1:
                chr_key = chr(ord_key) if ord_key <= 255 else chr(0)

                if chr_key >= 'A' and chr_key <= 'Z':
                    shift_pressed = True
                    ord_key += 32
                elif chr_key == '?':
                    shift_pressed = True
                    ord_key = ord('/')
                elif chr_key == '<':
                    shift_pressed = True
                    ord_key = ord(',')
                elif chr_key == '>':
                    shift_pressed = True
                    ord_key = ord('.')
        else:
            if sleep_time != 0:
                time.sleep(sleep_time)

        if has_capture_keys and ord_key != -1:
            self.add_key_event ( self.focus_wnd_name, ord_key, False, False, shift_pressed)

    def on_wait_any_key(self):
        cv2.waitKey(0)

class InteractColab(InteractBase):

    def is_support_windows(self):
        return False

    def is_colab(self):
        return True

    def on_destroy_all_windows(self):
        pass
        #clear_output()

    def on_create_window (self, wnd_name):
        pass
        #clear_output()

    def on_destroy_window (self, wnd_name):
        pass

    def on_show_image (self, wnd_name, img):
        pass
        # # cv2 stores colors as BGR; convert to RGB
        # if img.ndim == 3:
        #     if img.shape[2] == 4:
        #         img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        #     else:
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = PIL.Image.fromarray(img)
        # plt.imshow(img)
        # plt.show()

    def on_capture_mouse (self, wnd_name):
        pass
        #print("on_capture_mouse(): Colab does not support")

    def on_capture_keys (self, wnd_name):
        pass
        #print("on_capture_keys(): Colab does not support")

    def on_process_messages(self, sleep_time=0):
        time.sleep(sleep_time)

    def on_wait_any_key(self):
        pass
        #print("on_wait_any_key(): Colab does not support")

if is_colab:
    interact = InteractColab()
else:
    interact = InteractDesktop()
