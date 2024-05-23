from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class QIconDB():
    @staticmethod
    def initialize(icon_path):
        QIconDB.app_icon         = QIcon ( str(icon_path / 'app_icon.png') )
        QIconDB.delete_poly      = QIcon ( str(icon_path / 'delete_poly.png') )
        QIconDB.undo_pt          = QIcon ( str(icon_path / 'undo_pt.png') )
        QIconDB.redo_pt          = QIcon ( str(icon_path / 'redo_pt.png') )
        QIconDB.poly_color_red   = QIcon ( str(icon_path / 'poly_color_red.png') )
        QIconDB.poly_color_green = QIcon ( str(icon_path / 'poly_color_green.png') )
        QIconDB.poly_color_blue  = QIcon ( str(icon_path / 'poly_color_blue.png') )
        QIconDB.poly_type_include = QIcon ( str(icon_path / 'poly_type_include.png') )
        QIconDB.poly_type_exclude = QIcon ( str(icon_path / 'poly_type_exclude.png') )
        QIconDB.left  = QIcon ( str(icon_path / 'left.png') )
        QIconDB.right = QIcon ( str(icon_path / 'right.png') )
        QIconDB.trashcan = QIcon ( str(icon_path / 'trashcan.png') )
        QIconDB.pt_edit_mode = QIcon ( str(icon_path / 'pt_edit_mode.png') )
        QIconDB.view_lock_center = QIcon ( str(icon_path / 'view_lock_center.png') )
        QIconDB.view_baked = QIcon ( str(icon_path / 'view_baked.png') )
        QIconDB.view_xseg = QIcon ( str(icon_path / 'view_xseg.png') )
        QIconDB.view_xseg_overlay = QIcon ( str(icon_path / 'view_xseg_overlay.png') )
        