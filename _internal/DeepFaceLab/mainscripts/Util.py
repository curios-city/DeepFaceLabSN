import pickle
from pathlib import Path

import cv2

from DFLIMG import *
from facelib import LandmarksProcessor, FaceType
from core.interact import interact as io
from core import pathex
from core.cv2ex import *
from samplelib import PackedFaceset


def is_packed(input_path):
    if PackedFaceset.path_contains(input_path):
        io.log_info (f'\n{input_path} 包含打包的人脸集！请先解压它.\n')
        return True

def save_faceset_metadata_folder(input_path):
    # 将输入路径转换为 Path 对象
    input_path = Path(input_path)

    # 检查输入路径是否打包，如果是则返回
    if is_packed(input_path): 
        return

    # 定义元数据文件的路径
    metadata_filepath = input_path / 'meta.dat'

    # 记录保存元数据的信息
    io.log_info(f"将元数据保存至 {str(metadata_filepath)}\r\n")

    # 初始化一个空字典以存储元数据
    d = {}

    # 遍历输入路径中的图像文件
    for filepath in io.progress_bar_generator(pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)
        
        # 从图像文件加载 DFLIMG 对象
        dflimg = DFLIMG.load(filepath)

        # 检查 DFLIMG 是否有效且包含数据
        if dflimg is None or not dflimg.has_data():
            io.log_info(f"{filepath} 不是 DFL 图像文件")
            continue
            
        # 从 DFLIMG 获取元数据并存储在字典中
        dfl_dict = dflimg.get_dict()
        d[filepath.name] = (dflimg.get_shape(), dfl_dict)

    try:
        # 将包含元数据的字典写入元数据文件
        with open(metadata_filepath, "wb") as f:
            f.write(pickle.dumps(d))
    except:
        # 如果文件写入失败，则引发异常
        raise Exception('无法保存 %s' % (filename))

    # 记录关于编辑图像的信息
    io.log_info("现在您可以编辑图像.")
    io.log_info("!!! 保持文件夹中的相同文件名.")
    io.log_info("您可以更改图像的大小，还原过程将缩小回原始大小")
    io.log_info("之后，请使用还原元数据.")

def restore_faceset_metadata_folder(input_path):
    # 将输入路径转换为 Path 对象
    input_path = Path(input_path)

    # 检查输入路径是否打包，如果是则返回
    if is_packed(input_path):
        return

    # 定义元数据文件的路径
    metadata_filepath = input_path / 'meta.dat'
    
    # 记录关于恢复元数据的信息
    io.log_info(f"从{str(metadata_filepath)}恢复元数据.\r\n")

    # 如果元数据文件不存在，则记录错误
    if not metadata_filepath.exists():
        io.log_err(f"找不到{str(metadata_filepath)}.")

    try:
        # 从元数据文件读取包含元数据的字典
        with open(metadata_filepath, "rb") as f:
            d = pickle.loads(f.read())
    except:
        # 如果文件读取失败，则引发异常
        raise FileNotFoundError(filename)

    # 遍历具有特定扩展名的输入路径中的图像文件
    for filepath in io.progress_bar_generator(pathex.get_image_paths(input_path, image_extensions=['.jpg'], return_Path_class=True), "Processing"):
        # 获取当前文件的保存的元数据
        saved_data = d.get(filepath.name, None)
        
        # 检查当前文件是否存在保存的元数据
        if saved_data is None:
            io.log_info(f"{filepath}没有保存的元数据")
            continue
        
        # 从保存的元数据中提取形状和 DFL 字典
        shape, dfl_dict = saved_data

        # 使用 OpenCV 读取图像
        img = cv2_imread(filepath)

        # 如果图像形状与保存的形状不匹配，则调整图像大小
        if img.shape != shape:
            img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)

            # 以原始文件名保存调整大小后的图像
            cv2_imwrite(str(filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # 检查文件扩展名并相应地处理
        if filepath.suffix == '.jpg':
            # 加载 DFLJPG 对象并设置字典
            dflimg = DFLJPG.load(filepath)
            dflimg.set_dict(dfl_dict)
            dflimg.save()
        else:
            # 如果文件扩展名不是 '.jpg'，则跳过
            continue

    # 处理完成后删除元数据文件
    metadata_filepath.unlink()
    
    
def add_landmarks_debug_images(input_path):

    if is_packed(input_path) : return

    io.log_info ("添加标记点调试图像...")

    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        img = cv2_imread(str(filepath))

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是DFL图像文件")
            continue
        
        if img is not None:
            face_landmarks = dflimg.get_landmarks()
            face_type = FaceType.fromString ( dflimg.get_face_type() )
            
            if face_type == FaceType.MARK_ONLY:
                rect = dflimg.get_source_rect()
                LandmarksProcessor.draw_rect_landmarks(img, rect, face_landmarks, FaceType.FULL )
            else:
                LandmarksProcessor.draw_landmarks(img, face_landmarks, transparent_mask=True )
            
            
            
            output_file = '{}{}'.format( str(Path(str(input_path)) / filepath.stem),  '_debug.jpg')
            cv2_imwrite(output_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

def recover_original_aligned_filename(input_path):

    if is_packed(input_path) : return

    io.log_info ("恢复原始对齐后的文件名...")

    files = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是DFL图像文件")
            continue

        files += [ [filepath, None, dflimg.get_source_filename(), False] ]

    files_len = len(files)
    for i in io.progress_bar_generator( range(files_len), "Sorting" ):
        fp, _, sf, converted = files[i]

        if converted:
            continue

        sf_stem = Path(sf).stem

        files[i][1] = fp.parent / ( sf_stem + '_0' + fp.suffix )
        files[i][3] = True
        c = 1

        for j in range(i+1, files_len):
            fp_j, _, sf_j, converted_j = files[j]
            if converted_j:
                continue

            if sf_j == sf:
                files[j][1] = fp_j.parent / ( sf_stem + ('_%d' % (c)) + fp_j.suffix )
                files[j][3] = True
                c += 1

    for file in io.progress_bar_generator( files, "Renaming", leave=False ):
        fs, _, _, _ = file
        dst = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (dst)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )

    for file in io.progress_bar_generator( files, "Renaming" ):
        fs, fd, _, _ = file
        fs = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (fd)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )

def export_faceset_mask(input_dir):
    for filename in io.progress_bar_generator(pathex.get_image_paths (input_dir), "Processing"):
        filepath = Path(filename)

        if '_mask' in filepath.stem:
            continue

        mask_filepath = filepath.parent / (filepath.stem+'_mask'+filepath.suffix)

        dflimg = DFLJPG.load(filepath)

        H,W,C = dflimg.shape

        seg_ie_polys = dflimg.get_seg_ie_polys()

        if seg_ie_polys.has_polys():
            mask = np.zeros ((H,W,1), dtype=np.float32)
            seg_ie_polys.overlay_mask(mask)
        elif dflimg.has_xseg_mask():
            mask = dflimg.get_xseg_mask()
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
        else:
            raise Exception(f'no mask in file {filepath}')


        cv2_imwrite(mask_filepath, (mask*255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 100] )
