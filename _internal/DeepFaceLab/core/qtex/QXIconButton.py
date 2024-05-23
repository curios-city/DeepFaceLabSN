from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from localization import StringsDB
from .QXMainWindow import *

class QXIconButton(QPushButton):
    """
    Custom Icon button that works through keyEvent system, without shortcut of QAction
    works only with QXMainWindow as global window class
    currently works only with one-key shortcut
    """

    def __init__(self, icon, 
                    tooltip=None, 
                    shortcut=None,                    
                    click_func=None,                  
                    first_repeat_delay=300,
                    repeat_delay=20,
                    ):

        super().__init__(icon, "")

        self.setIcon(icon)
        
        if shortcut is not None:
            tooltip = f"{tooltip} ( {StringsDB['S_HOT_KEY'] }: {shortcut} )"
        
        self.setToolTip(tooltip)
            
        
        self.seq = QKeySequence(shortcut) if shortcut is not None else None
        
        QXMainWindow.inst.add_keyPressEvent_listener ( self.on_keyPressEvent )
        QXMainWindow.inst.add_keyReleaseEvent_listener ( self.on_keyReleaseEvent )
        
        self.click_func = click_func
        self.first_repeat_delay = first_repeat_delay
        self.repeat_delay = repeat_delay
        self.repeat_timer = None
        
        self.op_device = None
        
        self.pressed.connect( lambda : self.action(is_pressed=True)  )
        self.released.connect( lambda : self.action(is_pressed=False)  )
        
    def action(self, is_pressed=None, op_device=None):
        if self.click_func is None:
            return

        if is_pressed is not None:
            if is_pressed:
                if self.repeat_timer is None:
                    self.click_func()
                    self.repeat_timer = QTimer()
                    self.repeat_timer.timeout.connect(self.action)
                    self.repeat_timer.start(self.first_repeat_delay)
            else:
                if self.repeat_timer is not None:
                    self.repeat_timer.stop()
                    self.repeat_timer = None
        else:
            self.click_func()
            if self.repeat_timer is not None:
                self.repeat_timer.setInterval(self.repeat_delay)
        
    def on_keyPressEvent(self, ev):              
        key = ev.nativeVirtualKey()
        if ev.isAutoRepeat():
            return
            
        if self.seq is not None:
            if key == self.seq[0]:
                self.action(is_pressed=True)

    def on_keyReleaseEvent(self, ev):
        key = ev.nativeVirtualKey()
        if ev.isAutoRepeat():
            return
        if self.seq is not None:
            if key == self.seq[0]:
                self.action(is_pressed=False)
