import json
import os
import operator
import shutil
import traceback
from pathlib import Path
import numpy as np
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from DFLIMG import *
from facelib import XSegNet, LandmarksProcessor, FaceType
from samplelib import PackedFaceset
import pickle


def is_packed(input_path):
    if PackedFaceset.path_contains(input_path):
        io.log_info (f'\n{input_path} 包含打包的人脸集！请先解压\n')
        return True

def apply_xseg(input_path, model_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 未找到，请确保它存在.')

    if not model_path.exists():
        raise ValueError(f'{model_path} 未找到，请确保它存在.')

    if is_packed(input_path) : return
        
    face_type = None
    
    # 收集所有模型数据文件的名称和最后修改时间
    saved_models_names = []
    for filepath in pathex.get_file_paths(model_path):
        filepath_name = filepath.name
        if filepath_name.endswith(f'XSeg_data.dat'):
            # 如果文件名以模型类名结尾，将文件名和最后修改时间添加到列表中
            saved_models_names += [(filepath_name.split('_')[0], os.path.getmtime(filepath))]

    # 按修改时间倒序排序
    saved_models_names = sorted(saved_models_names, key=operator.itemgetter(1), reverse=True)
    saved_models_names = [x[0] for x in saved_models_names]

    # 如果有保存的模型
    if len(saved_models_names) == 1:
        model_name=saved_models_names[0]  #XSeg
        
    elif len(saved_models_names) > 1:

        io.log_info("选择一个模型")

        for i, model_name in enumerate(saved_models_names):
            s = f"[{i}] : {model_name} "
            if i == 0:
                s += "- 上次执行"
            io.log_info(s)

        # 用户输入选择的模型索引或操作（重命名或删除）
        inp = io.input_str(f"", "0", show_default_value=False)
        #初始化变量
        model_idx = -1
        try:
            model_idx = np.clip(int(inp), 0, len(saved_models_names) - 1)
        except:
            pass

        if model_idx == -1:
            #意味着用户输入的内容不能被转换为有效的整数，或者转换后的值不在合法的索引范围内
            model_name = inp
        else:
            # 根据用户选择的索引设置当前模型名称
            model_name = saved_models_names[model_idx]

    else:
        # 如果没有保存的模型，提示用户输入新模型的名称
        print("没有发现XSeg模型, 请下载或者训练")


    if model_name == "XSeg":
        model_dat = model_path / ('XSeg_data.dat')
    else:
        model_dat = model_path / (model_name+'_XSeg_data.dat')
        
    if model_dat.exists():
        dat = pickle.loads( model_dat.read_bytes() )
        dat_options = dat.get('options', None)
        if dat_options is not None:
            face_type = dat_options.get('face_type', None)
            if model_name == "XSeg":
                full_name= "XSeg"
                resolution = 256
            else:
                resolution = dat_options.get('resolution', None)
                full_name= model_name+'_XSeg'
        
    if face_type is None:
        face_type = io.input_str ("XSeg模型人脸类型 XSeg model face type", 'same', ['h','mf','f','wf','head','same', 'custom'], help_message="指定经过训练的XSeg模型的面部类型。例如，如果XSeg模型训练为WF，但facesset是HEAD，则指定WF仅对HEAD的WF部分应用XSeg。默认值 'same'").lower()
        if face_type == 'same':
            face_type = None
    
    if face_type is not None:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'custom' : FaceType.CUSTOM,
                     'head' : FaceType.HEAD}[face_type]
                     
    io.log_info(f'将训练好的 XSeg 模型应用于 {input_path.name}/ 文件夹.')

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)

    xseg = XSegNet(name=full_name, 
                    resolution=resolution,
                    load_weights=True,
                    weights_file_root=model_path,
                    data_format=nn.data_format,
                    raise_on_no_model_files=True)
    
    xseg_res = xseg.get_resolution()
              
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} 不是DFLIMG文件')
            continue
        
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
        
        img_face_type = FaceType.fromString( dflimg.get_face_type() )
        if face_type is not None and img_face_type != face_type or img_face_type == FaceType.CUSTOM: # custom always goes for eqvivalents
            lmrks = dflimg.get_source_landmarks()
            
            fmat = LandmarksProcessor.get_transform_mat(lmrks, w, face_type)
            imat = LandmarksProcessor.get_transform_mat(lmrks, w, img_face_type)
            
            g_p = LandmarksProcessor.transform_points (np.float32([(0,0),(w,0),(0,w) ]), fmat, True)
            g_p2 = LandmarksProcessor.transform_points (g_p, imat)
            
            mat = cv2.getAffineTransform( g_p2, np.float32([(0,0),(w,0),(0,w) ]) )
            
            img = cv2.warpAffine(img, mat, (w, w), cv2.INTER_LANCZOS4)
            img = cv2.resize(img, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        else:
            if w != xseg_res:
                img = cv2.resize( img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 )    
                    
        if len(img.shape) == 2:
            img = img[...,None]            
    
        mask = xseg.extract(img)
        
        if face_type is not None and img_face_type != face_type or img_face_type == FaceType.CUSTOM:
            mask = cv2.resize(mask, (w, w), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.warpAffine( mask, mat, (w,w), np.zeros( (h,w,c), dtype=np.float), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1    
        dflimg.set_xseg_mask(mask)
        dflimg.save()
        
def fetch_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 未找到，请确保它存在')

    if is_packed(input_path) : return
    
    output_path = input_path.parent / (input_path.name + '_xseg')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'将包含Xseg遮罩的人脸图片复制到 {output_path.name}/ 文件夹')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    
    files_copied = []
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} 不是 DFLIMG文件')
            continue
        
        ie_polys = dflimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied.append(filepath)
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'已复制文件数: {len(files_copied)}')
    
    is_delete = io.input_bool (f"\r\n删除原始文件?", True)
    if is_delete:
        for filepath in files_copied:
            Path(filepath).unlink()
               
def remove_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 未找到，请确保它存在')

    if is_packed(input_path) : return
    
    io.log_info(f'处理文件夹 {input_path}')

    io.input_str('按回车enter键继续.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} 不是 DFLIMG文件')
            continue
        
        if dflimg.has_xseg_mask():
            dflimg.set_xseg_mask(None)
            dflimg.save()
            files_processed += 1
    io.log_info(f'已处理文件数: {files_processed}')
    
def remove_xseg_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} 未找到，请确保它存在')

    if is_packed(input_path) : return
    
    io.log_info(f'处理文件夹 {input_path}')

    io.input_str('按回车enter键继续.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} 不是 DFLIMG 文件')
            continue

        if dflimg.has_seg_ie_polys():
            dflimg.set_seg_ie_polys(None)
            dflimg.save()            
            files_processed += 1
            
    io.log_info(f'已处理文件数: {files_processed}')
