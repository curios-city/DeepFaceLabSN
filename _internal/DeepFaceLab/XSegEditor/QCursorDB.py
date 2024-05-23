from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
                                       
class QCursorDB():
    @staticmethod
    def initialize(cursor_path):
        QCursorDB.cross_red = QCursor ( QPixmap ( str(cursor_path / 'cross_red.png') ) )
        QCursorDB.cross_green = QCursor ( QPixmap ( str(cursor_path / 'cross_green.png') ) )
        QCursorDB.cross_blue = QCursor ( QPixmap ( str(cursor_path / 'cross_blue.png') ) )
