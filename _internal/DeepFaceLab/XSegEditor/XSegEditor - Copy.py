import json
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
import traceback
from enum import IntEnum
from types import SimpleNamespace as sn

import cv2
import numpy as np
import numpy.linalg as npla
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from core import imagelib, pathex
from core.cv2ex import *
from core.imagelib import SegIEPoly, SegIEPolys, SegIEPolyType, sd
from core.qtex import *
from DFLIMG import *
from localization import StringsDB, system_language
from samplelib import PackedFaceset

from .QCursorDB import QCursorDB
from .QIconDB import QIconDB
from .QStringDB import QStringDB
from .QImageDB import QImageDB

class OpMode(IntEnum):
    NONE = 0
    DRAW_PTS = 1
    EDIT_PTS = 2
    VIEW_BAKED = 3
    VIEW_XSEG_MASK = 4
    VIEW_XSEG_OVERLAY_MASK = 5

class PTEditMode(IntEnum):
    MOVE = 0
    ADD_DEL = 1

class DragType(IntEnum):
    NONE = 0
    IMAGE_LOOK = 1
    POLY_PT = 2

class ViewLock(IntEnum):
    NONE = 0
    CENTER = 1

class QUIConfig():
    @staticmethod
    def initialize(icon_size = 48, icon_spacer_size=16, preview_bar_icon_size=64):
        QUIConfig.icon_q_size = QSize(icon_size, icon_size)
        QUIConfig.icon_spacer_q_size = QSize(icon_spacer_size, icon_spacer_size)
        QUIConfig.preview_bar_icon_q_size = QSize(preview_bar_icon_size, preview_bar_icon_size)

class ImagePreviewSequenceBar(QFrame):
    def __init__(self, preview_images_count, icon_size):
        super().__init__()
        self.preview_images_count = preview_images_count = max(1, preview_images_count + (preview_images_count % 2 -1) )

        self.icon_size = icon_size

        black_q_img = QImage(np.zeros( (icon_size,icon_size,3) ).data, icon_size, icon_size, 3*icon_size, QImage.Format_RGB888)
        self.black_q_pixmap = QPixmap.fromImage(black_q_img)

        self.image_containers = [ QLabel() for i in range(preview_images_count)]

        main_frame_l_cont_hl = QGridLayout()
        main_frame_l_cont_hl.setContentsMargins(0,0,0,0)
        #main_frame_l_cont_hl.setSpacing(0)



        for i in range(len(self.image_containers)):
            q_label = self.image_containers[i]
            q_label.setScaledContents(True)
            if i == preview_images_count//2:
                q_label.setMinimumSize(icon_size+16, icon_size+16 )
                q_label.setMaximumSize(icon_size+16, icon_size+16 )
            else:
                q_label.setMinimumSize(icon_size, icon_size )
                q_label.setMaximumSize(icon_size, icon_size )
                opacity_effect = QGraphicsOpacityEffect()
                opacity_effect.setOpacity(0.5)
                q_label.setGraphicsEffect(opacity_effect)

            q_label.setSizePolicy (QSizePolicy.Fixed, QSizePolicy.Fixed)

            main_frame_l_cont_hl.addWidget (q_label, 0, i)

        self.setLayout(main_frame_l_cont_hl)

        self.prev_img_conts = self.image_containers[(preview_images_count//2) -1::-1]
        self.next_img_conts = self.image_containers[preview_images_count//2:]

        self.update_images()

    def get_preview_images_count(self):
        return self.preview_images_count

    def update_images(self, prev_imgs=None, next_imgs=None):
        # Fix arrays
        if prev_imgs is None:
            prev_imgs = []
        prev_img_conts_len = len(self.prev_img_conts)
        prev_q_imgs_len = len(prev_imgs)
        if prev_q_imgs_len < prev_img_conts_len:
            for i in range ( prev_img_conts_len - prev_q_imgs_len ):
                prev_imgs.append(None)
        elif prev_q_imgs_len > prev_img_conts_len:
            prev_imgs = prev_imgs[:prev_img_conts_len]

        if next_imgs is None:
            next_imgs = []
        next_img_conts_len = len(self.next_img_conts)
        next_q_imgs_len = len(next_imgs)
        if next_q_imgs_len < next_img_conts_len:
            for i in range ( next_img_conts_len - next_q_imgs_len ):
                next_imgs.append(None)
        elif next_q_imgs_len > next_img_conts_len:
            next_imgs = next_imgs[:next_img_conts_len]

        for i,img in enumerate(prev_imgs):
            self.prev_img_conts[i].setPixmap( QPixmap.fromImage( QImage_from_np(img) ) if img is not None else self.black_q_pixmap )

        for i,img in enumerate(next_imgs):
            self.next_img_conts[i].setPixmap( QPixmap.fromImage( QImage_from_np(img) ) if img is not None else self.black_q_pixmap )

class ColorScheme():
    def __init__(self, unselected_color, selected_color, outline_color, outline_width, pt_outline_color, cross_cursor):
        self.poly_unselected_brush = QBrush(unselected_color)
        self.poly_selected_brush = QBrush(selected_color)

        self.poly_outline_solid_pen = QPen(outline_color, outline_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.poly_outline_dot_pen = QPen(outline_color, outline_width, Qt.DotLine, Qt.RoundCap, Qt.RoundJoin)

        self.pt_outline_pen = QPen(pt_outline_color)
        self.cross_cursor = cross_cursor

class CanvasConfig():

    def __init__(self,
                 pt_radius=4,
                 pt_select_radius=8,
                 color_schemes=None,
                 **kwargs):
        self.pt_radius = pt_radius
        self.pt_select_radius = pt_select_radius

        if color_schemes is None:
            color_schemes = [
                    ColorScheme( QColor(192,0,0,alpha=0), QColor(192,0,0,alpha=72), QColor(192,0,0), 2, QColor(255,255,255), QCursorDB.cross_red ),
                    ColorScheme( QColor(0,192,0,alpha=0), QColor(0,192,0,alpha=72), QColor(0,192,0), 2, QColor(255,255,255), QCursorDB.cross_green ),
                    ColorScheme( QColor(0,0,192,alpha=0), QColor(0,0,192,alpha=72), QColor(0,0,192), 2, QColor(255,255,255), QCursorDB.cross_blue ),
                    ]
        self.color_schemes = color_schemes

class QCanvasControlsLeftBar(QFrame):

    def __init__(self):
        super().__init__()
        #==============================================
        btn_poly_type_include = QToolButton()
        self.btn_poly_type_include_act = QActionEx( QIconDB.poly_type_include, QStringDB.btn_poly_type_include_tip, shortcut='Q', shortcut_in_tooltip=True, is_checkable=True)
        btn_poly_type_include.setDefaultAction(self.btn_poly_type_include_act)
        btn_poly_type_include.setIconSize(QUIConfig.icon_q_size)

        btn_poly_type_exclude = QToolButton()
        self.btn_poly_type_exclude_act = QActionEx( QIconDB.poly_type_exclude, QStringDB.btn_poly_type_exclude_tip, shortcut='W', shortcut_in_tooltip=True, is_checkable=True)
        btn_poly_type_exclude.setDefaultAction(self.btn_poly_type_exclude_act)
        btn_poly_type_exclude.setIconSize(QUIConfig.icon_q_size)

        self.btn_poly_type_act_grp = QActionGroup (self)
        self.btn_poly_type_act_grp.addAction(self.btn_poly_type_include_act)
        self.btn_poly_type_act_grp.addAction(self.btn_poly_type_exclude_act)
        self.btn_poly_type_act_grp.setExclusive(True)
        #==============================================
        btn_undo_pt = QToolButton()
        self.btn_undo_pt_act = QActionEx( QIconDB.undo_pt, QStringDB.btn_undo_pt_tip, shortcut='Ctrl+Z', shortcut_in_tooltip=True, is_auto_repeat=True)
        btn_undo_pt.setDefaultAction(self.btn_undo_pt_act)
        btn_undo_pt.setIconSize(QUIConfig.icon_q_size)

        btn_redo_pt = QToolButton()
        self.btn_redo_pt_act = QActionEx( QIconDB.redo_pt, QStringDB.btn_redo_pt_tip, shortcut='Ctrl+Shift+Z',  shortcut_in_tooltip=True, is_auto_repeat=True)
        btn_redo_pt.setDefaultAction(self.btn_redo_pt_act)
        btn_redo_pt.setIconSize(QUIConfig.icon_q_size)

        btn_delete_poly = QToolButton()
        self.btn_delete_poly_act = QActionEx( QIconDB.delete_poly, QStringDB.btn_delete_poly_tip, shortcut='Delete', shortcut_in_tooltip=True)
        btn_delete_poly.setDefaultAction(self.btn_delete_poly_act)
        btn_delete_poly.setIconSize(QUIConfig.icon_q_size)
        #==============================================
        btn_pt_edit_mode = QToolButton()
        self.btn_pt_edit_mode_act = QActionEx( QIconDB.pt_edit_mode, QStringDB.btn_pt_edit_mode_tip, shortcut_in_tooltip=True, is_checkable=True)
        btn_pt_edit_mode.setDefaultAction(self.btn_pt_edit_mode_act)
        btn_pt_edit_mode.setIconSize(QUIConfig.icon_q_size)
        #==============================================

        controls_bar_frame2_l = QVBoxLayout()
        controls_bar_frame2_l.addWidget ( btn_poly_type_include )
        controls_bar_frame2_l.addWidget ( btn_poly_type_exclude )
        controls_bar_frame2 = QFrame()
        controls_bar_frame2.setFrameShape(QFrame.StyledPanel)
        controls_bar_frame2.setSizePolicy (QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_bar_frame2.setLayout(controls_bar_frame2_l)

        controls_bar_frame3_l = QVBoxLayout()
        controls_bar_frame3_l.addWidget ( btn_undo_pt )
        controls_bar_frame3_l.addWidget ( btn_redo_pt )
        controls_bar_frame3_l.addWidget ( btn_delete_poly )
        controls_bar_frame3 = QFrame()
        controls_bar_frame3.setFrameShape(QFrame.StyledPanel)
        controls_bar_frame3.setSizePolicy (QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_bar_frame3.setLayout(controls_bar_frame3_l)

        controls_bar_frame4_l = QVBoxLayout()
        controls_bar_frame4_l.addWidget ( btn_pt_edit_mode )
        controls_bar_frame4 = QFrame()
        controls_bar_frame4.setFrameShape(QFrame.StyledPanel)
        controls_bar_frame4.setSizePolicy (QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_bar_frame4.setLayout(controls_bar_frame4_l)

        btn_view_lock_center = QToolButton()
        self.btn_view_lock_center_act = QActionEx( QIconDB.view_lock_center, QStringDB.btn_view_lock_center_tip, shortcut_in_tooltip=True, is_checkable=True)
        btn_view_lock_center.setDefaultAction(self.btn_view_lock_center_act)
        btn_view_lock_center.setIconSize(QUIConfig.icon_q_size)
        
        controls_bar_frame5_l = QVBoxLayout()
        controls_bar_frame5_l.addWidget ( btn_view_lock_center )
        controls_bar_frame5 = QFrame()
        controls_bar_frame5.setFrameShape(QFrame.StyledPanel)
        controls_bar_frame5.setSizePolicy (QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_bar_frame5.setLayout(controls_bar_frame5_l)
        
        
        controls_bar_l = QVBoxLayout()
        controls_bar_l.setContentsMargins(0,0,0,0)
        controls_bar_l.addWidget(controls_bar_frame2)
        controls_bar_l.addWidget(controls_bar_frame3)
        controls_bar_l.addWidget(controls_bar_frame4)        
        controls_bar_l.addWidget(controls_bar_frame5)
        
        self.setSizePolicy ( QSizePolicy.Fixed, QSizePolicy.Expanding )
        self.setLayout(controls_bar_l)

class QCanvasControlsRightBar(QFrame):

    def __init__(self):
        super().__init__()
        #==============================================
        btn_poly_color_red = QToolButton()
        self.btn_poly_color_red_act = QActionEx( QIconDB.poly_color_red, QStringDB.btn_poly_color_red_tip, shortcut='1', shortcut_in_tooltip=True, is_checkable=True)
        btn_poly_color_red.setDefaultAction(self.btn_poly_color_red_act)
        btn_poly_color_red.setIconSize(QUIConfig.icon_q_size)

        btn_poly_color_green = QToolButton()
        self.btn_poly_color_green_act = QActionEx( QIconDB.poly_color_green, QStringDB.btn_poly_color_green_tip, shortcut='2', shortcut_in_tooltip=True, is_checkable=True)
        btn_poly_color_green.setDefaultAction(self.btn_poly_color_green_act)
        btn_poly_color_green.setIconSize(QUIConfig.icon_q_size)

        btn_poly_color_blue = QToolButton()
        self.btn_poly_color_blue_act = QActionEx( QIconDB.poly_color_blue, QStringDB.btn_poly_color_blue_tip, shortcut='3', shortcut_in_tooltip=True, is_checkable=True)
        btn_poly_color_blue.setDefaultAction(self.btn_poly_color_blue_act)
        btn_poly_color_blue.setIconSize(QUIConfig.icon_q_size)

        btn_view_baked_mask = QToolButton()
        self.btn_view_baked_mask_act = QActionEx( QIconDB.view_baked, QStringDB.btn_view_baked_mask_tip, shortcut='4', shortcut_in_tooltip=True, is_checkable=True)
        btn_view_baked_mask.setDefaultAction(self.btn_view_baked_mask_act)
        btn_view_baked_mask.setIconSize(QUIConfig.icon_q_size)

        btn_view_xseg_mask = QToolButton()
        self.btn_view_xseg_mask_act = QActionEx( QIconDB.view_xseg, QStringDB.btn_view_xseg_mask_tip, shortcut='5', shortcut_in_tooltip=True, is_checkable=True)
        btn_view_xseg_mask.setDefaultAction(self.btn_view_xseg_mask_act)
        btn_view_xseg_mask.setIconSize(QUIConfig.icon_q_size)

        btn_view_xseg_overlay_mask = QToolButton()
        self.btn_view_xseg_overlay_mask_act = QActionEx( QIconDB.view_xseg_overlay, QStringDB.btn_view_xseg_overlay_mask_tip, shortcut='6', shortcut_in_tooltip=True, is_checkable=True)
        btn_view_xseg_overlay_mask.setDefaultAction(self.btn_view_xseg_overlay_mask_act)
        btn_view_xseg_overlay_mask.setIconSize(QUIConfig.icon_q_size)

        self.btn_poly_color_act_grp = QActionGroup (self)
        self.btn_poly_color_act_grp.addAction(self.btn_poly_color_red_act)
        self.btn_poly_color_act_grp.addAction(self.btn_poly_color_green_act)
        self.btn_poly_color_act_grp.addAction(self.btn_poly_color_blue_act)
        self.btn_poly_color_act_grp.addAction(self.btn_view_baked_mask_act)
        self.btn_poly_color_act_grp.addAction(self.btn_view_xseg_mask_act)
        self.btn_poly_color_act_grp.addAction(self.btn_view_xseg_overlay_mask_act)
        self.btn_poly_color_act_grp.setExclusive(True)
        #==============================================
        
        btn_xseg_to_poly = QToolButton()
        self.btn_xseg_to_poly_act = QActionEx( QIconDB.view_lock_center, QStringDB.btn_view_lock_center_tip, shortcut_in_tooltip=True, is_checkable=False)
        btn_xseg_to_poly.setDefaultAction(self.btn_xseg_to_poly_act)
        btn_xseg_to_poly.setIconSize(QUIConfig.icon_q_size)
        
        controls_bar_frame1_l = QVBoxLayout()
        controls_bar_frame1_l.addWidget ( btn_poly_color_red )
        controls_bar_frame1_l.addWidget ( btn_poly_color_green )
        controls_bar_frame1_l.addWidget ( btn_poly_color_blue )
        controls_bar_frame1_l.addWidget ( btn_view_baked_mask )
        controls_bar_frame1_l.addWidget ( btn_view_xseg_mask )
        controls_bar_frame1_l.addWidget ( btn_view_xseg_overlay_mask )
        controls_bar_frame1 = QFrame()
        controls_bar_frame1.setFrameShape(QFrame.StyledPanel)
        controls_bar_frame1.setSizePolicy (QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_bar_frame1.setLayout(controls_bar_frame1_l)

        controls_bar_frame2_l = QVBoxLayout()
        controls_bar_frame2_l.addWidget ( btn_xseg_to_poly )
        controls_bar_frame2 = QFrame()
        controls_bar_frame2.setFrameShape(QFrame.StyledPanel)
        controls_bar_frame2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_bar_frame2.setLayout(controls_bar_frame2_l)

        controls_bar_l = QVBoxLayout()
        controls_bar_l.setContentsMargins(0,0,0,0)
        controls_bar_l.addWidget(controls_bar_frame1)
        controls_bar_l.addWidget(controls_bar_frame2)
        
        self.setSizePolicy ( QSizePolicy.Fixed, QSizePolicy.Expanding )
        self.setLayout(controls_bar_l)

class QCanvasOperator(QWidget):
    def __init__(self, cbar):
        super().__init__()
        self.cbar = cbar

        self.set_cbar_disabled()

        self.cbar.btn_poly_color_red_act.triggered.connect ( lambda : self.set_color_scheme_id(0) )
        self.cbar.btn_poly_color_green_act.triggered.connect ( lambda : self.set_color_scheme_id(1) )
        self.cbar.btn_poly_color_blue_act.triggered.connect ( lambda : self.set_color_scheme_id(2) )
        self.cbar.btn_view_baked_mask_act.toggled.connect ( lambda : self.set_op_mode(OpMode.VIEW_BAKED) )
        self.cbar.btn_view_xseg_mask_act.toggled.connect ( self.set_view_xseg_mask )
        self.cbar.btn_view_xseg_overlay_mask_act.toggled.connect ( self.set_view_xseg_overlay_mask )

        self.cbar.btn_poly_type_include_act.triggered.connect ( lambda : self.set_poly_include_type(SegIEPolyType.INCLUDE) )
        self.cbar.btn_poly_type_exclude_act.triggered.connect ( lambda : self.set_poly_include_type(SegIEPolyType.EXCLUDE) )

        self.cbar.btn_undo_pt_act.triggered.connect ( lambda : self.action_undo_pt() )
        self.cbar.btn_redo_pt_act.triggered.connect ( lambda : self.action_redo_pt() )

        self.cbar.btn_delete_poly_act.triggered.connect ( lambda : self.action_delete_poly() )

        self.cbar.btn_pt_edit_mode_act.toggled.connect ( lambda is_checked: self.set_pt_edit_mode( PTEditMode.ADD_DEL if is_checked else PTEditMode.MOVE ) )
        self.cbar.btn_view_lock_center_act.toggled.connect ( lambda is_checked: self.set_view_lock( ViewLock.CENTER if is_checked else ViewLock.NONE ) )

        self.cbar.btn_xseg_to_poly_act.triggered.connect ( lambda : self.action_xseg_to_poly() )


        self.mouse_in_widget = False

        QXMainWindow.inst.add_keyPressEvent_listener ( self.on_keyPressEvent )
        QXMainWindow.inst.add_keyReleaseEvent_listener ( self.on_keyReleaseEvent )

        self.qp = QPainter()
        self.initialized = False
        self.last_state = None

    def initialize(self, img, img_look_pt=None, view_scale=None, ie_polys=None, xseg_mask=None, canvas_config=None ):
        self.img = img
        q_img = self.q_img = QImage_from_np(img)
        self.img_pixmap = QPixmap.fromImage(q_img)

        self.xseg_mask_in = imagelib.normalize_channels(xseg_mask, 1)
        self.xseg_mask_pixmap = None
        self.xseg_overlay_mask_pixmap = None
        
        if xseg_mask is not None:
            h,w,c = img.shape
            xseg_mask = cv2.resize(xseg_mask, (w,h), cv2.INTER_CUBIC)
            xseg_mask = imagelib.normalize_channels(xseg_mask, 1)
            xseg_img = img.astype(np.float32)/255.0
            xseg_overlay_mask = xseg_img*(1-xseg_mask)*0.5 + xseg_img*xseg_mask
            xseg_overlay_mask = np.clip(xseg_overlay_mask*255, 0, 255).astype(np.uint8)
            xseg_mask = np.clip(xseg_mask*255, 0, 255).astype(np.uint8)
            self.xseg_mask_pixmap = QPixmap.fromImage(QImage_from_np(xseg_mask))
            self.xseg_overlay_mask_pixmap = QPixmap.fromImage(QImage_from_np(xseg_overlay_mask))

        self.img_size = QSize_to_np (self.img_pixmap.size())

        self.img_look_pt = img_look_pt
        self.view_scale = view_scale

        if ie_polys is None:
            ie_polys = SegIEPolys()
        self.ie_polys = ie_polys

        if canvas_config is None:
            canvas_config = CanvasConfig()
        self.canvas_config = canvas_config

        # UI init
        self.set_cbar_disabled()
        self.cbar.btn_poly_color_act_grp.setDisabled(False)
        self.cbar.btn_poly_type_act_grp.setDisabled(False)

        # Initial vars
        self.current_cursor = None
        self.mouse_hull_poly = None
        self.mouse_wire_poly = None
        self.drag_type = DragType.NONE
        self.mouse_cli_pt = np.zeros((2,), np.float32 )

        # Initial state
        self.set_op_mode(OpMode.NONE)
        self.set_color_scheme_id(1)
        self.set_poly_include_type(SegIEPolyType.INCLUDE)
        self.set_pt_edit_mode(PTEditMode.MOVE)
        self.set_view_lock(ViewLock.NONE)

        # Apply last state
        if self.last_state is not None:
            self.set_color_scheme_id(self.last_state.color_scheme_id)
            if self.last_state.op_mode is not None:
                self.set_op_mode(self.last_state.op_mode)

        self.initialized = True

        self.setMouseTracking(True)
        self.update_cursor()
        self.update()


    def finalize(self):
        if self.initialized:
            if self.op_mode == OpMode.DRAW_PTS:
                self.set_op_mode(OpMode.EDIT_PTS)

            self.last_state = sn(op_mode = self.op_mode if self.op_mode in [OpMode.VIEW_BAKED, OpMode.VIEW_XSEG_MASK, OpMode.VIEW_XSEG_OVERLAY_MASK] else None,
                                 color_scheme_id = self.color_scheme_id,
                               )

            self.img_pixmap = None
            self.update_cursor(is_finalize=True)
            self.setMouseTracking(False)
            self.setFocusPolicy(Qt.NoFocus)
            self.set_cbar_disabled()
            self.initialized = False
            self.update()

    # ====================================================================================
    # ====================================================================================
    # ====================================== GETTERS =====================================
    # ====================================================================================
    # ====================================================================================

    def is_initialized(self):
        return self.initialized

    def get_ie_polys(self):
        return self.ie_polys

    def get_cli_center_pt(self):
        return np.round(QSize_to_np(self.size())/2.0)

    def get_img_look_pt(self):
        img_look_pt = self.img_look_pt
        if img_look_pt is None:
            img_look_pt = self.img_size / 2
        return img_look_pt

    def get_view_scale(self):
        view_scale = self.view_scale
        if view_scale is None:
            # Calc as scale to fit
            min_cli_size = np.min(QSize_to_np(self.size()))
            max_img_size = np.max(self.img_size)
            view_scale =  min_cli_size / max_img_size

        return view_scale

    def get_current_color_scheme(self):
        return self.canvas_config.color_schemes[self.color_scheme_id]

    def get_poly_pt_id_under_pt(self, poly, cli_pt):
        w = np.argwhere ( npla.norm ( cli_pt - self.img_to_cli_pt( poly.get_pts() ), axis=1 )  <= self.canvas_config.pt_select_radius )
        return None if len(w) == 0 else w[-1][0]

    def get_poly_edge_id_pt_under_pt(self, poly, cli_pt):
        cli_pts = self.img_to_cli_pt(poly.get_pts())
        if len(cli_pts) >= 3:
            edge_dists, projs = sd.dist_to_edges(cli_pts, cli_pt, is_closed=True)
            edge_id = np.argmin(edge_dists)
            dist = edge_dists[edge_id]
            pt = projs[edge_id]
            if dist <= self.canvas_config.pt_select_radius:
                return edge_id, pt
        return None, None

    def get_poly_by_pt_near_wire(self, cli_pt):
        pt_select_radius = self.canvas_config.pt_select_radius

        for poly in reversed(self.ie_polys.get_polys()):
            pts = poly.get_pts()
            if len(pts) >= 3:
                cli_pts = self.img_to_cli_pt(pts)

                edge_dists, _ = sd.dist_to_edges(cli_pts, cli_pt, is_closed=True)

                if np.min(edge_dists) <= pt_select_radius or \
                   any( npla.norm ( cli_pt - cli_pts, axis=1 ) <= pt_select_radius ):
                    return poly
        return None

    def get_poly_by_pt_in_hull(self, cli_pos):
        img_pos = self.cli_to_img_pt(cli_pos)

        for poly in reversed(self.ie_polys.get_polys()):
            pts = poly.get_pts()
            if len(pts) >= 3:
                if cv2.pointPolygonTest( pts, tuple(img_pos), False) >= 0:
                    return poly

        return None

    def img_to_cli_pt(self, p):
        return (p - self.get_img_look_pt()) * self.get_view_scale() + self.get_cli_center_pt()# QSize_to_np(self.size())/2.0

    def cli_to_img_pt(self, p):
        return (p - self.get_cli_center_pt() ) / self.get_view_scale() + self.get_img_look_pt()

    def img_to_cli_rect(self, rect):
        tl = QPoint_to_np(rect.topLeft())
        xy = self.img_to_cli_pt(tl)
        xy2 = self.img_to_cli_pt(tl + QSize_to_np(rect.size()) ) - xy
        return QRect ( *xy.astype(np.int), *xy2.astype(np.int) )

    # ====================================================================================
    # ====================================================================================
    # ====================================== SETTERS =====================================
    # ====================================================================================
    # ====================================================================================
    def set_op_mode(self, op_mode, op_poly=None):
        if not hasattr(self,'op_mode'):
            self.op_mode = None
            self.op_poly = None

        if self.op_mode != op_mode:
            # Finalize prev mode
            if self.op_mode == OpMode.NONE:
                self.cbar.btn_poly_type_act_grp.setDisabled(True)
            elif self.op_mode == OpMode.DRAW_PTS:
                self.cbar.btn_undo_pt_act.setDisabled(True)
                self.cbar.btn_redo_pt_act.setDisabled(True)
                self.cbar.btn_view_lock_center_act.setDisabled(True)
                # Reset view_lock when exit from DRAW_PTS
                self.set_view_lock(ViewLock.NONE)
                # Remove unfinished poly
                if self.op_poly.get_pts_count() < 3:
                    self.ie_polys.remove_poly(self.op_poly)

            elif self.op_mode == OpMode.EDIT_PTS:
                self.cbar.btn_pt_edit_mode_act.setDisabled(True)
                self.cbar.btn_delete_poly_act.setDisabled(True)
                # Reset pt_edit_move when exit from EDIT_PTS
                self.set_pt_edit_mode(PTEditMode.MOVE)
            elif self.op_mode == OpMode.VIEW_BAKED:
                self.cbar.btn_view_baked_mask_act.setChecked(False)
            elif self.op_mode == OpMode.VIEW_XSEG_MASK:
                self.cbar.btn_view_xseg_mask_act.setChecked(False)
                self.cbar.btn_xseg_to_poly_act.setDisabled(True)
            elif self.op_mode == OpMode.VIEW_XSEG_OVERLAY_MASK:
                self.cbar.btn_view_xseg_overlay_mask_act.setChecked(False)
                self.cbar.btn_xseg_to_poly_act.setDisabled(True)
            self.op_mode = op_mode

            # Initialize new mode
            if op_mode == OpMode.NONE:
                self.cbar.btn_poly_type_act_grp.setDisabled(False)
            elif op_mode == OpMode.DRAW_PTS:
                self.cbar.btn_undo_pt_act.setDisabled(False)
                self.cbar.btn_redo_pt_act.setDisabled(False)
                self.cbar.btn_view_lock_center_act.setDisabled(False)
            elif op_mode == OpMode.EDIT_PTS:
                self.cbar.btn_pt_edit_mode_act.setDisabled(False)
                self.cbar.btn_delete_poly_act.setDisabled(False)
            elif op_mode == OpMode.VIEW_BAKED:
                self.cbar.btn_view_baked_mask_act.setChecked(True )
                n = QImage_to_np ( self.q_img ).astype(np.float32) / 255.0
                h,w,c = n.shape
                mask = np.zeros( (h,w,1), dtype=np.float32 )
                self.ie_polys.overlay_mask(mask)
                n = (mask*255).astype(np.uint8)
                self.img_baked_pixmap = QPixmap.fromImage(QImage_from_np(n))
            elif op_mode == OpMode.VIEW_XSEG_MASK:
                self.cbar.btn_view_xseg_mask_act.setChecked(True)                
                if self.xseg_mask_in is not None:
                    self.cbar.btn_xseg_to_poly_act.setDisabled(False)
            elif op_mode == OpMode.VIEW_XSEG_OVERLAY_MASK:
                self.cbar.btn_view_xseg_overlay_mask_act.setChecked(True)
                if self.xseg_mask_in is not None:
                    self.cbar.btn_xseg_to_poly_act.setDisabled(False)
                    
            if op_mode in [OpMode.DRAW_PTS, OpMode.EDIT_PTS]:
                self.mouse_op_poly_pt_id = None
                self.mouse_op_poly_edge_id = None
                self.mouse_op_poly_edge_id_pt = None

            self.op_poly = op_poly
            if op_poly is not None:
                self.update_mouse_info()

            self.update_cursor()
            self.update()

    def set_pt_edit_mode(self, pt_edit_mode):
        if not hasattr(self, 'pt_edit_mode') or self.pt_edit_mode != pt_edit_mode:
            self.pt_edit_mode = pt_edit_mode
            self.update_cursor()
            self.update()
        self.cbar.btn_pt_edit_mode_act.setChecked( self.pt_edit_mode == PTEditMode.ADD_DEL )

    def set_view_lock(self, view_lock):
        if not hasattr(self, 'view_lock') or self.view_lock != view_lock:
            if hasattr(self, 'view_lock') and self.view_lock != view_lock:
                if view_lock == ViewLock.CENTER:
                    self.img_look_pt = self.mouse_img_pt
                    QCursor.setPos ( self.mapToGlobal( QPoint_from_np(self.img_to_cli_pt(self.img_look_pt)) ))

            self.view_lock = view_lock
            self.update()
        self.cbar.btn_view_lock_center_act.setChecked( self.view_lock == ViewLock.CENTER )

    def set_cbar_disabled(self):
        self.cbar.btn_delete_poly_act.setDisabled(True)
        self.cbar.btn_undo_pt_act.setDisabled(True)
        self.cbar.btn_redo_pt_act.setDisabled(True)
        self.cbar.btn_pt_edit_mode_act.setDisabled(True)
        self.cbar.btn_view_lock_center_act.setDisabled(True)
        self.cbar.btn_poly_color_act_grp.setDisabled(True)
        self.cbar.btn_poly_type_act_grp.setDisabled(True)
        self.cbar.btn_xseg_to_poly_act.setDisabled(True)

    def set_color_scheme_id(self, id):
        if self.op_mode == OpMode.VIEW_BAKED:
            self.set_op_mode(OpMode.NONE)

        if not hasattr(self, 'color_scheme_id') or self.color_scheme_id != id:
            self.color_scheme_id = id
            self.update_cursor()
            self.update()

        if self.color_scheme_id == 0:
            self.cbar.btn_poly_color_red_act.setChecked( True )
        elif self.color_scheme_id == 1:
            self.cbar.btn_poly_color_green_act.setChecked( True )
        elif self.color_scheme_id == 2:
            self.cbar.btn_poly_color_blue_act.setChecked( True )

    def set_poly_include_type(self, poly_include_type):
        if not hasattr(self, 'poly_include_type' ) or \
           ( self.poly_include_type != poly_include_type and \
             self.op_mode in [OpMode.NONE, OpMode.EDIT_PTS] ):
            self.poly_include_type = poly_include_type
            self.update()

        self.cbar.btn_poly_type_include_act.setChecked(self.poly_include_type == SegIEPolyType.INCLUDE)
        self.cbar.btn_poly_type_exclude_act.setChecked(self.poly_include_type == SegIEPolyType.EXCLUDE)

    def set_view_xseg_mask(self, is_checked):
        if is_checked:
            self.set_op_mode(OpMode.VIEW_XSEG_MASK)
        else:
            self.set_op_mode(OpMode.NONE)

        self.cbar.btn_view_xseg_mask_act.setChecked(is_checked )

    def set_view_xseg_overlay_mask(self, is_checked):
        if is_checked:
            self.set_op_mode(OpMode.VIEW_XSEG_OVERLAY_MASK)
        else:
            self.set_op_mode(OpMode.NONE)

        self.cbar.btn_view_xseg_overlay_mask_act.setChecked(is_checked )

    # ====================================================================================
    # ====================================================================================
    # ====================================== METHODS =====================================
    # ====================================================================================
    # ====================================================================================

    def update_cursor(self, is_finalize=False):
        if not self.initialized:
            return

        if not self.mouse_in_widget or is_finalize:
            if self.current_cursor is not None:
                QApplication.restoreOverrideCursor()
                self.current_cursor = None
        else:
            color_cc = self.get_current_color_scheme().cross_cursor
            nc = Qt.ArrowCursor

            if self.drag_type == DragType.IMAGE_LOOK:
                nc = Qt.ClosedHandCursor
            else:

                if self.op_mode == OpMode.NONE:
                    nc = color_cc
                    if self.mouse_wire_poly is not None:
                        nc = Qt.PointingHandCursor

                elif self.op_mode == OpMode.DRAW_PTS:
                    nc = color_cc
                elif self.op_mode == OpMode.EDIT_PTS:
                    nc = Qt.ArrowCursor

                    if self.mouse_op_poly_pt_id is not None:
                        nc = Qt.PointingHandCursor

                    if self.pt_edit_mode == PTEditMode.ADD_DEL:

                        if self.mouse_op_poly_edge_id is not None and \
                        self.mouse_op_poly_pt_id is None:
                            nc = color_cc
            if self.current_cursor != nc:
                if self.current_cursor is None:
                    QApplication.setOverrideCursor(nc)
                else:
                    QApplication.changeOverrideCursor(nc)
                self.current_cursor = nc

    def update_mouse_info(self, mouse_cli_pt=None):
        """
        Update selected polys/edges/points by given mouse position
        """
        if mouse_cli_pt is not None:
            self.mouse_cli_pt = mouse_cli_pt.astype(np.float32)

        self.mouse_img_pt = self.cli_to_img_pt(self.mouse_cli_pt)

        new_mouse_hull_poly = self.get_poly_by_pt_in_hull(self.mouse_cli_pt)

        if self.mouse_hull_poly != new_mouse_hull_poly:
            self.mouse_hull_poly = new_mouse_hull_poly
            self.update_cursor()
            self.update()

        new_mouse_wire_poly = self.get_poly_by_pt_near_wire(self.mouse_cli_pt)

        if self.mouse_wire_poly != new_mouse_wire_poly:
            self.mouse_wire_poly = new_mouse_wire_poly
            self.update_cursor()
            self.update()

        if self.op_mode in [OpMode.DRAW_PTS, OpMode.EDIT_PTS]:
            new_mouse_op_poly_pt_id = self.get_poly_pt_id_under_pt (self.op_poly, self.mouse_cli_pt)
            if self.mouse_op_poly_pt_id != new_mouse_op_poly_pt_id:
                self.mouse_op_poly_pt_id = new_mouse_op_poly_pt_id
                self.update_cursor()
                self.update()

            new_mouse_op_poly_edge_id,\
            new_mouse_op_poly_edge_id_pt = self.get_poly_edge_id_pt_under_pt (self.op_poly, self.mouse_cli_pt)
            if self.mouse_op_poly_edge_id != new_mouse_op_poly_edge_id:
                self.mouse_op_poly_edge_id = new_mouse_op_poly_edge_id
                self.update_cursor()
                self.update()

            if (self.mouse_op_poly_edge_id_pt.__class__ != new_mouse_op_poly_edge_id_pt.__class__) or \
               (isinstance(self.mouse_op_poly_edge_id_pt, np.ndarray) and \
                all(self.mouse_op_poly_edge_id_pt != new_mouse_op_poly_edge_id_pt)):

                self.mouse_op_poly_edge_id_pt = new_mouse_op_poly_edge_id_pt
                self.update_cursor()
                self.update()


    def action_undo_pt(self):
        if self.drag_type == DragType.NONE:
            if self.op_mode == OpMode.DRAW_PTS:
                if self.op_poly.undo() == 0:
                    self.ie_polys.remove_poly (self.op_poly)
                    self.set_op_mode(OpMode.NONE)
                self.update()

    def action_redo_pt(self):
        if self.drag_type == DragType.NONE:
            if self.op_mode == OpMode.DRAW_PTS:
                self.op_poly.redo()
                self.update()

    def action_delete_poly(self):
        if self.op_mode == OpMode.EDIT_PTS and \
            self.drag_type == DragType.NONE and \
            self.pt_edit_mode == PTEditMode.MOVE:
            # Delete current poly
            self.ie_polys.remove_poly (self.op_poly)
            self.set_op_mode(OpMode.NONE)

    def action_xseg_to_poly(self):
        
        cnts = cv2.findContours( (self.xseg_mask_in*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)        
        # Sort by countour area      
        cnts = sorted(cnts[0], key = cv2.contourArea, reverse = True)
        if len(cnts) != 0:            
            h,w,c = self.img.shape
            mh,mw,mc = self.xseg_mask_in.shape
            
            dh = h / mh
            dw = w / mw
            
            new_poly = self.ie_polys.add_poly(SegIEPolyType.INCLUDE)
            for pt in cnts[0].squeeze():            
                new_poly.add_pt( pt[0]*dw, pt[1]*dh )
            
            self.set_op_mode(OpMode.EDIT_PTS, op_poly=new_poly)
            
        
    # ====================================================================================
    # ====================================================================================
    # ================================== OVERRIDE QT METHODS =============================
    # ====================================================================================
    # ====================================================================================
    def on_keyPressEvent(self, ev):
        if not self.initialized:
            return
        key = ev.key()
        key_mods = int(ev.modifiers())
        if self.op_mode == OpMode.DRAW_PTS:
            self.set_view_lock(ViewLock.CENTER if key_mods == Qt.ShiftModifier else ViewLock.NONE )
        elif self.op_mode == OpMode.EDIT_PTS:
            self.set_pt_edit_mode(PTEditMode.ADD_DEL if key_mods == Qt.ControlModifier else PTEditMode.MOVE )

    def on_keyReleaseEvent(self, ev):
        if not self.initialized:
            return
        key = ev.key()
        key_mods = int(ev.modifiers())
        if self.op_mode == OpMode.DRAW_PTS:
            self.set_view_lock(ViewLock.CENTER if key_mods == Qt.ShiftModifier else ViewLock.NONE )
        elif self.op_mode == OpMode.EDIT_PTS:
            self.set_pt_edit_mode(PTEditMode.ADD_DEL if key_mods == Qt.ControlModifier else PTEditMode.MOVE )

    def enterEvent(self, ev):
        super().enterEvent(ev)
        self.mouse_in_widget = True
        self.update_cursor()

    def leaveEvent(self, ev):
        super().leaveEvent(ev)
        self.mouse_in_widget = False
        self.update_cursor()

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if not self.initialized:
            return

        self.update_mouse_info(QPoint_to_np(ev.pos()))

        btn = ev.button()

        if btn == Qt.LeftButton:
            if self.op_mode == OpMode.NONE:
                # Clicking in NO OPERATION mode
                if self.mouse_wire_poly is not None:
                    # Click on wire on any poly -> switch to EDIT_MODE
                    self.set_op_mode(OpMode.EDIT_PTS, op_poly=self.mouse_wire_poly)
                else:
                    # Click on empty space -> create new poly with one point
                    new_poly = self.ie_polys.add_poly(self.poly_include_type)
                    self.ie_polys.sort()
                    new_poly.add_pt(*self.mouse_img_pt)
                    self.set_op_mode(OpMode.DRAW_PTS, op_poly=new_poly )

            elif self.op_mode == OpMode.DRAW_PTS:
                # Clicking in DRAW_PTS mode
                if len(self.op_poly.get_pts()) >= 3 and self.mouse_op_poly_pt_id == 0:
                    # Click on first point -> close poly and switch to edit mode
                    self.set_op_mode(OpMode.EDIT_PTS, op_poly=self.op_poly)
                else:
                    # Click on empty space -> add point to current poly
                    self.op_poly.add_pt(*self.mouse_img_pt)
                    self.update()

            elif self.op_mode == OpMode.EDIT_PTS:
                # Clicking in EDIT_PTS mode

                if self.mouse_op_poly_pt_id is not None:
                    # Click on point of op_poly
                    if self.pt_edit_mode == PTEditMode.ADD_DEL:
                        # with mode -> delete point
                        self.op_poly.remove_pt(self.mouse_op_poly_pt_id)
                        if self.op_poly.get_pts_count() < 3:
                            # not enough points -> remove poly
                            self.ie_polys.remove_poly (self.op_poly)
                            self.set_op_mode(OpMode.NONE)
                        self.update()

                    elif self.drag_type == DragType.NONE:
                        # otherwise -> start drag
                        self.drag_type = DragType.POLY_PT
                        self.drag_cli_pt     = self.mouse_cli_pt
                        self.drag_poly_pt_id = self.mouse_op_poly_pt_id
                        self.drag_poly_pt    = self.op_poly.get_pts()[ self.drag_poly_pt_id ]
                elif self.mouse_op_poly_edge_id is not None:
                    # Click on edge of op_poly
                    if self.pt_edit_mode == PTEditMode.ADD_DEL:
                        # with mode -> insert new point
                        edge_img_pt = self.cli_to_img_pt(self.mouse_op_poly_edge_id_pt)
                        self.op_poly.insert_pt (self.mouse_op_poly_edge_id+1, edge_img_pt)
                        self.update()
                    else:
                        # Otherwise do nothing
                        pass
                else:
                    # other cases -> unselect poly
                    self.set_op_mode(OpMode.NONE)

        elif btn == Qt.MiddleButton:
            if self.drag_type == DragType.NONE:
                # Start image drag
                self.drag_type = DragType.IMAGE_LOOK
                self.drag_cli_pt = self.mouse_cli_pt
                self.drag_img_look_pt = self.get_img_look_pt()
                self.update_cursor()


    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if not self.initialized:
            return

        self.update_mouse_info(QPoint_to_np(ev.pos()))

        btn = ev.button()

        if btn == Qt.LeftButton:
            if self.op_mode == OpMode.EDIT_PTS:
                if self.drag_type == DragType.POLY_PT:
                    self.drag_type = DragType.NONE
                    self.update()

        elif btn == Qt.MiddleButton:
            if self.drag_type == DragType.IMAGE_LOOK:
                self.drag_type = DragType.NONE
                self.update_cursor()
                self.update()

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if not self.initialized:
            return

        prev_mouse_cli_pt = self.mouse_cli_pt
        self.update_mouse_info(QPoint_to_np(ev.pos()))

        if self.view_lock == ViewLock.CENTER:
            if npla.norm(self.mouse_cli_pt - prev_mouse_cli_pt) >= 1:
                self.img_look_pt = self.mouse_img_pt
                QCursor.setPos ( self.mapToGlobal( QPoint_from_np(self.img_to_cli_pt(self.img_look_pt)) ))

            self.update()

        if self.drag_type == DragType.IMAGE_LOOK:
            delta_pt = self.cli_to_img_pt(self.mouse_cli_pt) - self.cli_to_img_pt(self.drag_cli_pt)
            self.img_look_pt = self.drag_img_look_pt - delta_pt
            self.update()

        if self.op_mode == OpMode.DRAW_PTS:
            self.update()
        elif self.op_mode == OpMode.EDIT_PTS:
            if self.drag_type == DragType.POLY_PT:
                delta_pt = self.cli_to_img_pt(self.mouse_cli_pt) - self.cli_to_img_pt(self.drag_cli_pt)
                self.op_poly.set_point(self.drag_poly_pt_id, self.drag_poly_pt + delta_pt)
                self.update()

    def wheelEvent(self, ev):
        super().wheelEvent(ev)

        if not self.initialized:
            return

        mods = int(ev.modifiers())
        delta = ev.angleDelta()

        cli_pt = QPoint_to_np(ev.pos())

        if self.drag_type == DragType.NONE:
            sign = np.sign( delta.y() )
            prev_img_pos = self.cli_to_img_pt (cli_pt)
            delta_scale = sign*0.2 + sign * self.get_view_scale() / 10.0
            self.view_scale = np.clip(self.get_view_scale() + delta_scale, 1.0, 20.0)
            new_img_pos = self.cli_to_img_pt (cli_pt)
            if sign > 0:
                self.img_look_pt = self.get_img_look_pt() + (prev_img_pos-new_img_pos)#*1.5
            else:
                QCursor.setPos ( self.mapToGlobal(QPoint_from_np(self.img_to_cli_pt(prev_img_pos))) )
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.initialized:
            return

        qp = self.qp
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setRenderHint(QPainter.HighQualityAntialiasing)
        qp.setRenderHint(QPainter.SmoothPixmapTransform)

        src_rect = QRect(0, 0, *self.img_size)
        dst_rect = self.img_to_cli_rect( src_rect )

        if self.op_mode == OpMode.VIEW_BAKED:
            qp.drawPixmap(dst_rect, self.img_baked_pixmap, src_rect)
        elif self.op_mode == OpMode.VIEW_XSEG_MASK:
            if self.xseg_mask_pixmap is not None:
                qp.drawPixmap(dst_rect, self.xseg_mask_pixmap, src_rect)
        elif self.op_mode == OpMode.VIEW_XSEG_OVERLAY_MASK:
            if self.xseg_overlay_mask_pixmap is not None:
                qp.drawPixmap(dst_rect, self.xseg_overlay_mask_pixmap, src_rect)
        else:
            if self.img_pixmap is not None:
                qp.drawPixmap(dst_rect, self.img_pixmap, src_rect)

            polys = self.ie_polys.get_polys()
            polys_len = len(polys)

            color_scheme = self.get_current_color_scheme()

            pt_rad = self.canvas_config.pt_radius
            pt_rad_x2 = pt_rad*2

            pt_select_radius = self.canvas_config.pt_select_radius

            op_mode = self.op_mode
            op_poly = self.op_poly

            for i,poly in enumerate(polys):

                selected_pt_path = QPainterPath()
                poly_line_path = QPainterPath()
                pts_line_path = QPainterPath()

                pt_remove_cli_pt = None
                poly_pts = poly.get_pts()
                for pt_id, img_pt in enumerate(poly_pts):
                    cli_pt = self.img_to_cli_pt(img_pt)
                    q_cli_pt = QPoint_from_np(cli_pt)

                    if pt_id == 0:
                        poly_line_path.moveTo(q_cli_pt)
                    else:
                        poly_line_path.lineTo(q_cli_pt)


                    if poly == op_poly:
                        if self.op_mode == OpMode.DRAW_PTS or \
                            (self.op_mode == OpMode.EDIT_PTS and \
                            (self.pt_edit_mode == PTEditMode.MOVE) or \
                            (self.pt_edit_mode == PTEditMode.ADD_DEL and self.mouse_op_poly_pt_id == pt_id) \
                            ):
                            pts_line_path.moveTo( QPoint_from_np(cli_pt + np.float32([0,-pt_rad])) )
                            pts_line_path.lineTo( QPoint_from_np(cli_pt + np.float32([0,pt_rad])) )
                            pts_line_path.moveTo( QPoint_from_np(cli_pt + np.float32([-pt_rad,0])) )
                            pts_line_path.lineTo( QPoint_from_np(cli_pt + np.float32([pt_rad,0])) )

                        if (self.op_mode == OpMode.EDIT_PTS and \
                            self.pt_edit_mode == PTEditMode.ADD_DEL and \
                            self.mouse_op_poly_pt_id == pt_id):
                            pt_remove_cli_pt = cli_pt

                        if self.op_mode == OpMode.DRAW_PTS and \
                            len(op_poly.get_pts()) >= 3 and pt_id == 0 and self.mouse_op_poly_pt_id == pt_id:
                            # Circle around poly point
                            selected_pt_path.addEllipse(q_cli_pt, pt_rad_x2, pt_rad_x2)


                if poly == op_poly:
                    if op_mode == OpMode.DRAW_PTS:
                        # Line from last point to mouse
                        poly_line_path.lineTo( QPoint_from_np(self.mouse_cli_pt) )

                    if self.mouse_op_poly_pt_id is not None:
                        pass

                    if self.mouse_op_poly_edge_id_pt is not None:
                        if self.pt_edit_mode == PTEditMode.ADD_DEL and self.mouse_op_poly_pt_id is None:
                            # Ready to insert point on edge
                            m_cli_pt = self.mouse_op_poly_edge_id_pt
                            pts_line_path.moveTo( QPoint_from_np(m_cli_pt + np.float32([0,-pt_rad])) )
                            pts_line_path.lineTo( QPoint_from_np(m_cli_pt + np.float32([0,pt_rad])) )
                            pts_line_path.moveTo( QPoint_from_np(m_cli_pt + np.float32([-pt_rad,0])) )
                            pts_line_path.lineTo( QPoint_from_np(m_cli_pt + np.float32([pt_rad,0])) )

                if len(poly_pts) >= 2:
                    # Closing poly line
                    poly_line_path.lineTo( QPoint_from_np(self.img_to_cli_pt(poly_pts[0])) )

                # Draw calls
                qp.setPen(color_scheme.pt_outline_pen)
                qp.setBrush(QBrush())
                qp.drawPath(selected_pt_path)

                qp.setPen(color_scheme.poly_outline_solid_pen)
                qp.setBrush(QBrush())
                qp.drawPath(pts_line_path)

                if poly.get_type() == SegIEPolyType.INCLUDE:
                    qp.setPen(color_scheme.poly_outline_solid_pen)
                else:
                    qp.setPen(color_scheme.poly_outline_dot_pen)

                qp.setBrush(color_scheme.poly_unselected_brush)
                if op_mode == OpMode.NONE:
                    if poly == self.mouse_wire_poly:
                        qp.setBrush(color_scheme.poly_selected_brush)
                #else:
                #    if poly == op_poly:
                #        qp.setBrush(color_scheme.poly_selected_brush)

                qp.drawPath(poly_line_path)

                if pt_remove_cli_pt is not None:
                    qp.setPen(color_scheme.poly_outline_solid_pen)
                    qp.setBrush(QBrush())

                    qp.drawLine( *(pt_remove_cli_pt + np.float32([-pt_rad_x2,-pt_rad_x2])), *(pt_remove_cli_pt + np.float32([pt_rad_x2,pt_rad_x2])) )
                    qp.drawLine( *(pt_remove_cli_pt + np.float32([-pt_rad_x2,pt_rad_x2])), *(pt_remove_cli_pt + np.float32([pt_rad_x2,-pt_rad_x2])) )

        qp.end()

class QCanvas(QFrame):
    def __init__(self):
        super().__init__()

        self.canvas_control_left_bar = QCanvasControlsLeftBar()
        self.canvas_control_right_bar = QCanvasControlsRightBar()

        cbar = sn( btn_poly_color_red_act   = self.canvas_control_right_bar.btn_poly_color_red_act,
                   btn_poly_color_green_act = self.canvas_control_right_bar.btn_poly_color_green_act,
                   btn_poly_color_blue_act  = self.canvas_control_right_bar.btn_poly_color_blue_act,
                   btn_view_baked_mask_act  = self.canvas_control_right_bar.btn_view_baked_mask_act,
                   btn_view_xseg_mask_act  = self.canvas_control_right_bar.btn_view_xseg_mask_act,
                   btn_view_xseg_overlay_mask_act  = self.canvas_control_right_bar.btn_view_xseg_overlay_mask_act,
                   btn_poly_color_act_grp = self.canvas_control_right_bar.btn_poly_color_act_grp,
                   btn_xseg_to_poly_act = self.canvas_control_right_bar.btn_xseg_to_poly_act,

                   btn_poly_type_include_act = self.canvas_control_left_bar.btn_poly_type_include_act,
                   btn_poly_type_exclude_act = self.canvas_control_left_bar.btn_poly_type_exclude_act,
                   btn_poly_type_act_grp = self.canvas_control_left_bar.btn_poly_type_act_grp,
                   btn_undo_pt_act = self.canvas_control_left_bar.btn_undo_pt_act,
                   btn_redo_pt_act = self.canvas_control_left_bar.btn_redo_pt_act,
                   btn_delete_poly_act = self.canvas_control_left_bar.btn_delete_poly_act,
                   btn_pt_edit_mode_act = self.canvas_control_left_bar.btn_pt_edit_mode_act,                   
                   btn_view_lock_center_act = self.canvas_control_left_bar.btn_view_lock_center_act, )

        self.op = QCanvasOperator(cbar)
        self.l = QHBoxLayout()
        self.l.setContentsMargins(0,0,0,0)
        self.l.addWidget(self.canvas_control_left_bar)
        self.l.addWidget(self.op)
        self.l.addWidget(self.canvas_control_right_bar)
        self.setLayout(self.l)

class LoaderQSubprocessor(QSubprocessor):
    def __init__(self, image_paths, q_label, q_progressbar, on_finish_func ):

        self.image_paths = image_paths
        self.image_paths_len = len(image_paths)
        self.idxs = [*range(self.image_paths_len)]

        self.filtered_image_paths = self.image_paths.copy()

        self.image_paths_has_ie_polys = { image_path : False for image_path in self.image_paths }

        self.q_label = q_label
        self.q_progressbar = q_progressbar
        self.q_progressbar.setRange(0, self.image_paths_len)
        self.q_progressbar.setValue(0)
        self.q_progressbar.update()
        self.on_finish_func = on_finish_func
        self.done_count = 0
        super().__init__('LoaderQSubprocessor', LoaderQSubprocessor.Cli, 60)

    def get_data(self, host_dict):
        if len (self.idxs) > 0:
            idx = self.idxs.pop(0)
            image_path = self.image_paths[idx]
            self.q_label.setText(f'{QStringDB.loading_tip}... {image_path.name}')

            return idx, image_path

        return None

    def on_clients_finalized(self):
        self.on_finish_func([x for x in self.filtered_image_paths if x is not None], self.image_paths_has_ie_polys)

    def on_data_return (self, host_dict, data):
        self.idxs.insert(0, data[0])

    def on_result (self, host_dict, data, result):
        idx, has_dflimg, has_ie_polys = result

        if not has_dflimg:
            self.filtered_image_paths[idx] = None
        self.image_paths_has_ie_polys[self.image_paths[idx]] = has_ie_polys

        self.done_count += 1
        if self.q_progressbar is not None:
            self.q_progressbar.setValue(self.done_count)

    class Cli(QSubprocessor.Cli):
        def process_data(self, data):
            idx, filename = data
            dflimg = DFLIMG.load(filename)
            if dflimg is not None and dflimg.has_data():
                ie_polys = dflimg.get_seg_ie_polys()

                return idx, True, ie_polys.has_polys()
            return idx, False, False

class MainWindow(QXMainWindow):

    def __init__(self, input_dirpath, cfg_root_path):
        self.loading_frame = None
        self.help_frame = None

        super().__init__()

        self.input_dirpath = input_dirpath
        self.cfg_root_path = cfg_root_path

        self.cfg_path = cfg_root_path / 'MainWindow_cfg.dat'
        self.cfg_dict = pickle.loads(self.cfg_path.read_bytes()) if self.cfg_path.exists() else {}

        self.cached_images = {}
        self.cached_has_ie_polys = {}

        self.initialize_ui()

        # Loader
        self.loading_frame = QFrame(self.main_canvas_frame)
        self.loading_frame.setAutoFillBackground(True)
        self.loading_frame.setFrameShape(QFrame.StyledPanel)
        self.loader_label = QLabel()
        self.loader_progress_bar = QProgressBar()

        intro_image = QLabel()
        intro_image.setPixmap( QPixmap.fromImage(QImageDB.intro) )

        intro_image_frame_l = QVBoxLayout()
        intro_image_frame_l.addWidget(intro_image, alignment=Qt.AlignCenter)
        intro_image_frame = QFrame()
        intro_image_frame.setSizePolicy (QSizePolicy.Expanding, QSizePolicy.Expanding)
        intro_image_frame.setLayout(intro_image_frame_l)

        loading_frame_l = QVBoxLayout()
        loading_frame_l.addWidget (intro_image_frame)
        loading_frame_l.addWidget (self.loader_label)
        loading_frame_l.addWidget (self.loader_progress_bar)
        self.loading_frame.setLayout(loading_frame_l)

        self.loader_subprocessor = LoaderQSubprocessor( image_paths=pathex.get_image_paths(input_dirpath, return_Path_class=True),
                                                        q_label=self.loader_label,
                                                        q_progressbar=self.loader_progress_bar,
                                                        on_finish_func=self.on_loader_finish )


    def on_loader_finish(self, image_paths, image_paths_has_ie_polys):
        self.image_paths_done = []
        self.image_paths = image_paths
        self.image_paths_has_ie_polys = image_paths_has_ie_polys
        self.set_has_ie_polys_count ( len([ 1 for x in self.image_paths_has_ie_polys if self.image_paths_has_ie_polys[x] == True]) )
        self.loading_frame.hide()
        self.loading_frame = None

        self.process_next_image(first_initialization=True)

    def closeEvent(self, ev):
        self.cfg_dict['geometry'] = self.saveGeometry().data()
        self.cfg_path.write_bytes( pickle.dumps(self.cfg_dict) )


    def update_cached_images (self, count=5):
        d = self.cached_images

        for image_path in self.image_paths_done[:-count]+self.image_paths[count:]:
            if image_path in d:
                del d[image_path]

        for image_path in self.image_paths[:count]+self.image_paths_done[-count:]:
            if image_path not in d:
                img = cv2_imread(image_path)
                if img is not None:
                    d[image_path] = img

    def load_image(self, image_path):
        try:
            img = self.cached_images.get(image_path, None)
            if img is None:
                img = cv2_imread(image_path)
                self.cached_images[image_path] = img
            if img is None:
                io.log_err(f'Unable to load {image_path}')
        except:
            img = None

        return img

    def update_preview_bar(self):
        count = self.image_bar.get_preview_images_count()
        d = self.cached_images
        prev_imgs = [ d.get(image_path, None) for image_path in self.image_paths_done[-1:-count:-1] ]
        next_imgs = [ d.get(image_path, None) for image_path in self.image_paths[:count] ]
        self.image_bar.update_images(prev_imgs, next_imgs)


    def canvas_initialize(self, image_path, only_has_polys=False):
        if only_has_polys and not self.image_paths_has_ie_polys[image_path]:
            return False

        dflimg = DFLIMG.load(image_path)
        if not dflimg or not dflimg.has_data():
            return False

        ie_polys = dflimg.get_seg_ie_polys()
        xseg_mask = dflimg.get_xseg_mask()
        img = self.load_image(image_path)
        if img is None:
            return False

        self.canvas.op.initialize ( img,  ie_polys=ie_polys, xseg_mask=xseg_mask )

        self.filename_label.setText(f"{image_path.name}")

        return True

    def canvas_finalize(self, image_path):
        self.canvas.op.finalize()

        if image_path.exists():
            dflimg = DFLIMG.load(image_path)
            ie_polys = dflimg.get_seg_ie_polys()
            new_ie_polys = self.canvas.op.get_ie_polys()

            if not new_ie_polys.identical(ie_polys):
                new_has_ie_polys = new_ie_polys.has_polys()
                self.set_has_ie_polys_count ( self.get_has_ie_polys_count() + (1 if new_has_ie_polys else -1) )
                self.image_paths_has_ie_polys[image_path] = new_has_ie_polys
                dflimg.set_seg_ie_polys( new_ie_polys )
                dflimg.save()

        self.filename_label.setText(f"")

    def process_prev_image(self):
        key_mods = QApplication.keyboardModifiers()
        step = 5 if key_mods == Qt.ShiftModifier else 1
        only_has_polys = key_mods == Qt.ControlModifier

        if self.canvas.op.is_initialized():
            self.canvas_finalize(self.image_paths[0])

        while True:
            for _ in range(step):
                if len(self.image_paths_done) != 0:
                    self.image_paths.insert (0, self.image_paths_done.pop(-1))
                else:
                    break
            if len(self.image_paths) == 0:
                break

            ret = self.canvas_initialize(self.image_paths[0], len(self.image_paths_done) != 0 and only_has_polys)

            if ret or len(self.image_paths_done) == 0:
                break

        self.update_cached_images()
        self.update_preview_bar()

    def process_next_image(self, first_initialization=False):
        key_mods = QApplication.keyboardModifiers()

        step = 0 if first_initialization else 5 if key_mods == Qt.ShiftModifier else 1
        only_has_polys = False if first_initialization else key_mods == Qt.ControlModifier

        if self.canvas.op.is_initialized():
            self.canvas_finalize(self.image_paths[0])

        while True:
            for _ in range(step):
                if len(self.image_paths) != 0:
                    self.image_paths_done.append(self.image_paths.pop(0))
                else:
                    break
            if len(self.image_paths) == 0:
                break
            if self.canvas_initialize(self.image_paths[0], only_has_polys):
                break

        self.update_cached_images()
        self.update_preview_bar()

    def initialize_ui(self):

        self.canvas = QCanvas()

        image_bar = self.image_bar = ImagePreviewSequenceBar(preview_images_count=9, icon_size=QUIConfig.preview_bar_icon_q_size.width())
        image_bar.setSizePolicy ( QSizePolicy.Fixed, QSizePolicy.Fixed )


        btn_prev_image = QXIconButton(QIconDB.left, QStringDB.btn_prev_image_tip, shortcut='A', click_func=self.process_prev_image)
        btn_prev_image.setIconSize(QUIConfig.preview_bar_icon_q_size)

        btn_next_image = QXIconButton(QIconDB.right, QStringDB.btn_next_image_tip, shortcut='D', click_func=self.process_next_image)
        btn_next_image.setIconSize(QUIConfig.preview_bar_icon_q_size)


        preview_image_bar_frame_l = QHBoxLayout()
        preview_image_bar_frame_l.setContentsMargins(0,0,0,0)
        preview_image_bar_frame_l.addWidget ( btn_prev_image, alignment=Qt.AlignCenter)
        preview_image_bar_frame_l.addWidget ( image_bar)
        preview_image_bar_frame_l.addWidget ( btn_next_image, alignment=Qt.AlignCenter)

        preview_image_bar_frame = QFrame()
        preview_image_bar_frame.setSizePolicy ( QSizePolicy.Fixed, QSizePolicy.Fixed )
        preview_image_bar_frame.setLayout(preview_image_bar_frame_l)

        preview_image_bar_l = QHBoxLayout()
        preview_image_bar_l.addWidget (preview_image_bar_frame)

        preview_image_bar = QFrame()
        preview_image_bar.setFrameShape(QFrame.StyledPanel)
        preview_image_bar.setSizePolicy ( QSizePolicy.Expanding, QSizePolicy.Fixed )
        preview_image_bar.setLayout(preview_image_bar_l)

        label_font = QFont('Courier New')
        self.filename_label = QLabel()
        self.filename_label.setFont(label_font)

        self.has_ie_polys_count_label = QLabel()

        status_frame_l = QHBoxLayout()
        status_frame_l.setContentsMargins(0,0,0,0)
        status_frame_l.addWidget ( QLabel(), alignment=Qt.AlignCenter)
        status_frame_l.addWidget (self.filename_label, alignment=Qt.AlignCenter)
        status_frame_l.addWidget (self.has_ie_polys_count_label, alignment=Qt.AlignCenter)
        status_frame = QFrame()
        status_frame.setLayout(status_frame_l)

        main_canvas_l = QVBoxLayout()
        main_canvas_l.setContentsMargins(0,0,0,0)
        main_canvas_l.addWidget (self.canvas)
        main_canvas_l.addWidget (status_frame)
        main_canvas_l.addWidget (preview_image_bar)

        self.main_canvas_frame = QFrame()
        self.main_canvas_frame.setLayout(main_canvas_l)

        self.main_l = QHBoxLayout()
        self.main_l.setContentsMargins(0,0,0,0)
        self.main_l.addWidget (self.main_canvas_frame)

        self.setLayout(self.main_l)

        geometry = self.cfg_dict.get('geometry', None)
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.move( QPoint(0,0))

    def get_has_ie_polys_count(self):
        return self.has_ie_polys_count

    def set_has_ie_polys_count(self, c):
        self.has_ie_polys_count = c
        self.has_ie_polys_count_label.setText(f"{c} {QStringDB.labeled_tip}")

    def resizeEvent(self, ev):
        if self.loading_frame is not None:
            self.loading_frame.resize( ev.size() )
        if self.help_frame is not None:
            self.help_frame.resize( ev.size() )

def start(input_dirpath):
    """
    returns exit_code
    """
    io.log_info("Running XSeg editor.")

    if PackedFaceset.path_contains(input_dirpath):
        io.log_info (f'\n{input_dirpath} contains packed faceset! Unpack it first.\n')
        return 1

    root_path = Path(__file__).parent
    cfg_root_path = Path(tempfile.gettempdir())

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication([])
    app.setApplicationName("XSegEditor")
    app.setStyle('Fusion')

    QFontDatabase.addApplicationFont( str(root_path / 'gfx' / 'fonts' / 'NotoSans-Medium.ttf') )

    app.setFont( QFont('NotoSans'))

    QUIConfig.initialize()
    QStringDB.initialize()

    QIconDB.initialize( root_path / 'gfx' / 'icons' )
    QCursorDB.initialize( root_path / 'gfx' / 'cursors' )
    QImageDB.initialize( root_path / 'gfx' / 'images' )

    app.setWindowIcon(QIconDB.app_icon)
    app.setPalette( QDarkPalette() )

    win = MainWindow( input_dirpath=input_dirpath, cfg_root_path=cfg_root_path)

    win.show()
    win.raise_()

    app.exec_()
    return 0
