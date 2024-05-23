import numpy as np
import copy

from facelib import FaceType
from core.interact import interact as io


class MergerConfig(object):
    TYPE_NONE = 0
    TYPE_MASKED = 1
    TYPE_FACE_AVATAR = 2
    ####

    TYPE_IMAGE = 3
    TYPE_IMAGE_WITH_LANDMARKS = 4

    def __init__(self, type=0,
                       sharpen_mode=0,
                       blursharpen_amount=0,
                       **kwargs
                       ):
        self.type = type

        self.sharpen_dict = dict({0:"None", 1:'box', 2:'gaussian'})#, 3:'unsharpen'}

        #default changeable params
        self.sharpen_mode = sharpen_mode
        self.blursharpen_amount = blursharpen_amount

    def copy(self):
        return copy.copy(self)

    #overridable
    def ask_settings(self):
        s = """Choose sharpen mode: \n"""
        for key in self.sharpen_dict.keys():
            s += f"""({key}) {self.sharpen_dict[key]}\n"""
        io.log_info(s)
        self.sharpen_mode = io.input_int ("", 0, valid_list=self.sharpen_dict.keys(), help_message="Enhance details by applying sharpen filter.")

        if self.sharpen_mode != 0:
            self.blursharpen_amount = np.clip ( io.input_int ("Choose blur/sharpen amount", 0, add_info="-100..100"), -100, 100 )

    def toggle_sharpen_mode(self):
        a = list( self.sharpen_dict.keys() )
        self.sharpen_mode = a[ (a.index(self.sharpen_mode)+1) % len(a) ]

    def add_blursharpen_amount(self, diff):
        self.blursharpen_amount = np.clip ( self.blursharpen_amount+diff, -100, 100)

    #overridable
    def get_config(self):
        d = self.__dict__.copy()
        d.pop('type')
        return d

    #overridable
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, MergerConfig):
            return self.sharpen_mode == other.sharpen_mode and \
                   self.blursharpen_amount == other.blursharpen_amount

        return False

    #overridable
    def to_string(self, filename):
        r = ""
        r += f"sharpen_mode : {self.sharpen_dict[self.sharpen_mode]}\n"
        r += f"blursharpen_amount : {self.blursharpen_amount}\n"
        return r

mode_dict = {0:'original',
             1:'overlay',
             2:'hist-match',
             3:'seamless',
             4:'seamless-hist-match',
             5:'raw-rgb',
             6:'raw-predict'}

mode_str_dict = { mode_dict[key] : key for key in mode_dict.keys() }

mask_mode_dict = {0:'full',
                  1:'dst',
                  2:'learned-prd',
                  3:'learned-dst',
                  4:'learned-prd*learned-dst',
                  5:'learned-prd+learned-dst',
                  6:'XSeg-prd',
                  7:'XSeg-dst',
                  8:'XSeg-prd*XSeg-dst',
                  9:'XSeg-prd+XSeg-dst',
                  10:'learned-prd*learned-dst*XSeg-prd*XSeg-dst'
                  }


ctm_dict = { 0: "None", 1:"rct", 2:"lct", 3:"mkl", 4:"mkl-m", 5:"idt", 6:"idt-m", 7:"sot-m", 8:"mix-m" }
ctm_str_dict = {None:0, "rct":1, "lct":2, "mkl":3, "mkl-m":4, "idt":5, "idt-m":6, "sot-m":7, "mix-m":8 }

pre_sharpen_dict = dict({0:"None", 1:'gaussian'})#, 2:'unsharpen_mask'}
two_pass_dict = dict({0:"None", 1:'face', 2:'face+mask'})

class MergerConfigMasked(MergerConfig):

    def __init__(self, face_type=FaceType.FULL,
                       default_mode = 'overlay',
                       mode='overlay',
                       masked_hist_match=True,
                       hist_match_threshold = 238,
                       mask_mode = 4,
                       erode_mask_modifier = 0,
                       blur_mask_modifier = 0,
                       motion_blur_power = 0,
                       output_face_scale = 0,
                       super_resolution_power = 0,
                       color_transfer_mode = ctm_str_dict['rct'],
                       image_denoise_power = 0,
                       bicubic_degrade_power = 0,
                       color_degrade_power = 0,
                       pre_sharpen_power = 0,
                       pre_sharpen_mode=0,
                       two_pass_mode = 0, 
                       morph_power = 100,
                       is_morphable = False,
                       debug_mode = False,
                       **kwargs
                       ):

        super().__init__(type=MergerConfig.TYPE_MASKED, **kwargs)

        self.face_type = face_type
        if self.face_type not in [FaceType.HALF, FaceType.MID_FULL, FaceType.FULL, FaceType.WHOLE_FACE, FaceType.HEAD, FaceType.CUSTOM ]:
            raise ValueError("MergerConfigMasked does not support this type of face.")

        self.default_mode = default_mode

        #default changeable params
        if mode not in mode_str_dict:
            mode = mode_dict[1]

        self.mode = mode
        self.masked_hist_match = masked_hist_match
        self.hist_match_threshold = hist_match_threshold
        self.mask_mode = mask_mode
        self.erode_mask_modifier = erode_mask_modifier
        self.blur_mask_modifier = blur_mask_modifier
        self.motion_blur_power = motion_blur_power
        self.output_face_scale = output_face_scale
        self.super_resolution_power = super_resolution_power
        self.color_transfer_mode = color_transfer_mode
        self.image_denoise_power = image_denoise_power
        self.bicubic_degrade_power = bicubic_degrade_power
        self.color_degrade_power = color_degrade_power
        self.two_pass_mode = two_pass_mode
        self.pre_sharpen_power = pre_sharpen_power
        self.pre_sharpen_mode = pre_sharpen_mode
        self.morph_power = morph_power
        self.is_morphable = is_morphable
        self.debug_mode = debug_mode

    def copy(self):
        return copy.copy(self)

    def set_mode (self, mode):
        self.mode = mode_dict.get (mode, self.default_mode)

    def toggle_masked_hist_match(self):
        if self.mode == 'hist-match':
            self.masked_hist_match = not self.masked_hist_match
            
    def toggle_two_pass_mode(self):
        a = list( two_pass_dict.keys() )
        self.two_pass_mode = a[ (a.index(self.two_pass_mode)+1) % len(a) ]
        
    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        
                
    def toggle_sharpen_mode_multi(self, pre_sharpen=False):
        if pre_sharpen:
            self.toggle_sharpen_mode_presharpen()
        else:
            self.toggle_sharpen_mode()
            
    def toggle_sharpen_mode_presharpen(self):
        a = list( pre_sharpen_dict.keys() )
        self.pre_sharpen_mode = a[ (a.index(self.pre_sharpen_mode)+1) % len(a) ]
        
        
    def add_hist_match_threshold(self, diff):
        if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
            self.hist_match_threshold = np.clip ( self.hist_match_threshold+diff , 0, 255)

    def toggle_mask_mode(self):
        a = list( mask_mode_dict.keys() )
        self.mask_mode = a[ (a.index(self.mask_mode)+1) % len(a) ]

    def add_erode_mask_modifier(self, diff):
        self.erode_mask_modifier = np.clip ( self.erode_mask_modifier+diff , -400, 400)

    def add_blur_mask_modifier(self, diff):
        self.blur_mask_modifier = np.clip ( self.blur_mask_modifier+diff , 0, 400)

    def add_motion_blur_power(self, diff):
        self.motion_blur_power = np.clip ( self.motion_blur_power+diff, 0, 100)

    def add_output_face_scale(self, diff):
        self.output_face_scale = np.clip ( self.output_face_scale+diff , -50, 50)

    def toggle_color_transfer_mode(self):
        self.color_transfer_mode = (self.color_transfer_mode+1) % ( max(ctm_dict.keys())+1 )

    def add_super_resolution_power(self, diff):
        self.super_resolution_power = np.clip ( self.super_resolution_power+diff , 0, 100)

    def add_color_degrade_power(self, diff):
        self.color_degrade_power = np.clip ( self.color_degrade_power+diff , 0, 100)

    def add_image_denoise_power(self, diff):
        self.image_denoise_power = np.clip ( self.image_denoise_power+diff, 0, 500)

    def add_bicubic_degrade_power(self, diff):
        self.bicubic_degrade_power = np.clip ( self.bicubic_degrade_power+diff, 0, 100)
        
    def add_pre_sharpen_power(self, diff):
        self.pre_sharpen_power = np.clip ( self.pre_sharpen_power+diff, 0, 200)
        
    def add_morph_power(self, diff):
        if self.is_morphable:
            self.morph_power = np.clip ( self.morph_power+diff , 0, 100)

    def ask_settings(self):
        s = """Choose mode: \n"""
        for key in mode_dict.keys():
            s += f"""({key}) {mode_dict[key]}\n"""
        io.log_info(s)
        mode = io.input_int ("", mode_str_dict.get(self.default_mode, 1) )

        self.mode = mode_dict.get (mode, self.default_mode )

        if 'raw' not in self.mode:
            if self.mode == 'hist-match':
                self.masked_hist_match = io.input_bool("Masked hist match?", True)

            if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip ( io.input_int("Hist match threshold", 255, add_info="0..255"), 0, 255)

        s = """Choose mask mode: \n"""
        for key in mask_mode_dict.keys():
            s += f"""({key}) {mask_mode_dict[key]}\n"""
        io.log_info(s)
        self.mask_mode = io.input_int ("", 1, valid_list=mask_mode_dict.keys() )

        if 'raw' not in self.mode:
            self.erode_mask_modifier = np.clip ( io.input_int ("Choose erode mask modifier", 0, add_info="-400..400"), -400, 400)
            self.blur_mask_modifier =  np.clip ( io.input_int ("Choose blur mask modifier", 0, add_info="0..400"), 0, 400)
            self.motion_blur_power = np.clip ( io.input_int ("Choose motion blur power", 0, add_info="0..100"), 0, 100)
        

        s = """Choose two pass mode: \n"""
        for key in two_pass_dict.keys():
            s += f"""({key}) {two_pass_dict[key]}\n"""
        io.log_info(s)
        self.two_pass_mode = io.input_int ("", 0, valid_list=two_pass_dict.keys() )

        self.pre_sharpen_power = np.clip (io.input_int ("Choose pre_sharpen power", 0, help_message="Can enhance results by pre sharping before feeding it to the network.", add_info="0..100" ), 0, 200)
        
        if self.is_morphable:
            self.morph_power = np.clip (io.input_int ("Choose morph_power for moprhable models", 100, add_info="0..100" ), 0, 100)


        self.output_face_scale = np.clip (io.input_int ("Choose output face scale modifier", 0, add_info="-50..50" ), -50, 50)

        if 'raw' not in self.mode:
            self.color_transfer_mode = io.input_str ( "Color transfer to predicted face", None, valid_list=list(ctm_str_dict.keys())[1:] )
            self.color_transfer_mode = ctm_str_dict[self.color_transfer_mode]

        super().ask_settings()

        self.super_resolution_power = np.clip ( io.input_int ("Choose super resolution power", 0, add_info="0..100", help_message="Enhance details by applying superresolution network."), 0, 100)

        if 'raw' not in self.mode:
            self.image_denoise_power = np.clip ( io.input_int ("Choose image degrade by denoise power", 0, add_info="0..500"), 0, 500)
            self.bicubic_degrade_power = np.clip ( io.input_int ("Choose image degrade by bicubic rescale power", 0, add_info="0..100"), 0, 100)
            self.color_degrade_power = np.clip (  io.input_int ("Degrade color power of final image", 0, add_info="0..100"), 0, 100)
        
            self.debug_mode = io.input_bool("Debug mode?", False, help_message="Shows raw model output in the left and model input (dst face) in the left")

        io.log_info ("")

    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, MergerConfigMasked):
            return super().__eq__(other) and \
                   self.mode == other.mode and \
                   self.masked_hist_match == other.masked_hist_match and \
                   self.hist_match_threshold == other.hist_match_threshold and \
                   self.mask_mode == other.mask_mode and \
                   self.erode_mask_modifier == other.erode_mask_modifier and \
                   self.blur_mask_modifier == other.blur_mask_modifier and \
                   self.motion_blur_power == other.motion_blur_power and \
                   self.output_face_scale == other.output_face_scale and \
                   self.color_transfer_mode == other.color_transfer_mode and \
                   self.super_resolution_power == other.super_resolution_power and \
                   self.image_denoise_power == other.image_denoise_power and \
                   self.bicubic_degrade_power == other.bicubic_degrade_power and \
                   self.color_degrade_power == other.color_degrade_power and \
                   self.pre_sharpen_power == other.pre_sharpen_power and \
                   self.pre_sharpen_mode == other.pre_sharpen_mode and \
                   self.two_pass_mode == other.two_pass_mode and \
                   self.morph_power == other.morph_power and \
                   self.is_morphable == other.is_morphable and \
                   self.debug_mode == other.debug_mode 

        return False

    def to_string(self, filename):
        r = (
            f"""MergerConfig {filename}:\n"""
            f"""Mode: {self.mode}\n"""
            )

        if self.mode == 'hist-match':
            r += f"""masked_hist_match: {self.masked_hist_match}\n"""

        if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
            r += f"""hist_match_threshold: {self.hist_match_threshold}\n"""

        r += f"""mask_mode: { mask_mode_dict[self.mask_mode] }\n"""

        if 'raw' not in self.mode:
            r += (f"""erode_mask_modifier: {self.erode_mask_modifier}\n"""
                  f"""blur_mask_modifier: {self.blur_mask_modifier}\n"""
                  f"""motion_blur_power: {self.motion_blur_power}\n""")

        r += f"""output_face_scale: {self.output_face_scale}\n"""

        if 'raw' not in self.mode:
            r += f"""color_transfer_mode: {ctm_dict[self.color_transfer_mode]}\n"""
            r += super().to_string(filename)

        r += f"""super_resolution_power: {self.super_resolution_power}\n"""

        if 'raw' not in self.mode:
            r += (f"""image_denoise_power: {self.image_denoise_power}\n"""
                  f"""bicubic_degrade_power: {self.bicubic_degrade_power}\n"""
                  f"""color_degrade_power: {self.color_degrade_power}\n""")
        
       
        r += f"""pre_sharpen_mode: {pre_sharpen_dict[self.pre_sharpen_mode]}\n"""
        r += f"""pre_sharpen_power: {self.pre_sharpen_power}\n"""
        
        r += f"""two_pass-mode: {two_pass_dict[self.two_pass_mode]}\n"""
        if self.is_morphable:
            r += f"""morph_power: {self.morph_power}\n"""
        #r += f"""is_morphable: {self.is_morphable}\n"""
        r += f"""debug_mode: {self.debug_mode}\n"""
        
        r += "================"

        return r


class MergerConfigFaceAvatar(MergerConfig):

    def __init__(self, temporal_face_count=0,
                       add_source_image=False):
        super().__init__(type=MergerConfig.TYPE_FACE_AVATAR)
        self.temporal_face_count = temporal_face_count

        #changeable params
        self.add_source_image = add_source_image

    def copy(self):
        return copy.copy(self)

    #override
    def ask_settings(self):
        self.add_source_image = io.input_bool("Add source image?", False, help_message="Add source image for comparison.")
        super().ask_settings()

    def toggle_add_source_image(self):
        self.add_source_image = not self.add_source_image

    #override
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, MergerConfigFaceAvatar):
            return super().__eq__(other) and \
                   self.add_source_image == other.add_source_image

        return False

    #override
    def to_string(self, filename):
        return (f"MergerConfig {filename}:\n"
                f"add_source_image : {self.add_source_image}\n") + \
                super().to_string(filename) + "================"

