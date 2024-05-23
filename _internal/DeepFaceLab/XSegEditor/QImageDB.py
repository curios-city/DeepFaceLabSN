from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class QImageDB():
    @staticmethod
    def initialize(image_path):
        QImageDB.intro = QImage ( str(image_path / 'intro.png') )
