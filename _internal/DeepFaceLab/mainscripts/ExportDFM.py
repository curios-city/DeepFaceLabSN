import os
import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
import cv2
import models
from core.interact import interact as io


def main(model_class_name, saved_models_path):
    model = models.import_model(model_class_name)(
                        is_exporting=True,
                        saved_models_path=saved_models_path,
                        cpu_only=True)
    model.export_dfm () 
