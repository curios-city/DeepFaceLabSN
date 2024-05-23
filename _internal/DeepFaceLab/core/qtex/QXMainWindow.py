from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class QXMainWindow(QWidget):
    """
    Custom mainwindow class that provides global single instance and event listeners
    """
    inst = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        if QXMainWindow.inst is not None:
            raise Exception("QXMainWindow can only be one.")        
        QXMainWindow.inst = self
        
        self.keyPressEvent_listeners = []
        self.keyReleaseEvent_listeners = []
        self.setFocusPolicy(Qt.WheelFocus)
        
    def add_keyPressEvent_listener(self, func):
        self.keyPressEvent_listeners.append (func)
        
    def add_keyReleaseEvent_listener(self, func):
        self.keyReleaseEvent_listeners.append (func)
        
    def keyPressEvent(self, ev):
        super().keyPressEvent(ev)        
        for func in self.keyPressEvent_listeners:
            func(ev)
            
    def keyReleaseEvent(self, ev):
        super().keyReleaseEvent(ev)        
        for func in self.keyReleaseEvent_listeners:
            func(ev)