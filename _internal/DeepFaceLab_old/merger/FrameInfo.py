from pathlib import Path

class FrameInfo(object):
    def __init__(self, filepath=None, landmarks_list=None):
        self.filepath = filepath
        self.landmarks_list = landmarks_list or []
        self.motion_deg = 0
        self.motion_power = 0