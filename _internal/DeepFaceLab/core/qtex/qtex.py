import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from localization import StringsDB

from .QXMainWindow import *
            
                
class QActionEx(QAction):
    def __init__(self, icon, text, shortcut=None, trigger_func=None, shortcut_in_tooltip=False, is_checkable=False, is_auto_repeat=False ):
        super().__init__(icon, text)
        if shortcut is not None:
            self.setShortcut(shortcut)
            if shortcut_in_tooltip:
                
                self.setToolTip( f"{text} ( {StringsDB['S_HOT_KEY'] }: {shortcut} )")
                
        if trigger_func is not None:
            self.triggered.connect(trigger_func)
        if is_checkable:            
            self.setCheckable(True)
        self.setAutoRepeat(is_auto_repeat)
            
def QImage_from_np(img):
    if img.dtype != np.uint8:
        raise ValueError("img should be in np.uint8 format")
        
    h,w,c = img.shape
    if c == 1:
        fmt = QImage.Format_Grayscale8
    elif c == 3:
        fmt = QImage.Format_BGR888
    elif c == 4:
        fmt = QImage.Format_ARGB32
    else:
      raise ValueError("unsupported channel count")  
    
    return QImage(img.data, w, h, c*w, fmt )
        
def QImage_to_np(q_img, fmt=QImage.Format_BGR888):
    q_img = q_img.convertToFormat(fmt)

    width = q_img.width()
    height = q_img.height()
    
    b = q_img.constBits()
    b.setsize(height * width * 3)
    arr = np.frombuffer(b, np.uint8).reshape((height, width, 3))
    return arr#[::-1]
        
def QPixmap_from_np(img):    
    return QPixmap.fromImage(QImage_from_np(img))
    
def QPoint_from_np(n):
    return QPoint(*n.astype(np.int))
    
def QPoint_to_np(q):
    return np.int32( [q.x(), q.y()] )
    
def QSize_to_np(q):
    return np.int32( [q.width(), q.height()] )
    
class QDarkPalette(QPalette):
    def __init__(self):
        super().__init__()
        text_color = QColor(200,200,200)
        self.setColor(QPalette.Window, QColor(53, 53, 53))
        self.setColor(QPalette.WindowText, text_color )
        self.setColor(QPalette.Base, QColor(25, 25, 25))
        self.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ToolTipBase, text_color )
        self.setColor(QPalette.ToolTipText, text_color )
        self.setColor(QPalette.Text, text_color ) 
        self.setColor(QPalette.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ButtonText, Qt.white)
        self.setColor(QPalette.BrightText, Qt.red)
        self.setColor(QPalette.Link, QColor(42, 130, 218))
        self.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.HighlightedText, Qt.black)