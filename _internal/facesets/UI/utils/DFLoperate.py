# -*- coding: utf-8 -*-

import os
from UI.utils.DFLIMG.DFLJPG import DFLJPG
import cv2
from UI.utils import util
import glob
from pathlib import Path
from os import scandir

###----------- DFLJPG 判断 ----------------------##
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

ALIGNED_EXTENSIONS = ['.jpg', '.JPG']
FACEPAK_EXTENSIONS = ['.pak']

def isImgFile(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def isImgFile_New(strpath):
    if os.path.isfile(strpath) and os.path.getsize(strpath)!=0:
        return any(strpath.endswith(extension) for extension in IMG_EXTENSIONS)
    return False

def isDFLAlignedImg(imgFile:str)->bool:
    endswith_tuple = tuple(ALIGNED_EXTENSIONS)
    if imgFile.endswith(endswith_tuple):
        InputDflImg = DFLJPG.load(imgFile)
        if not InputDflImg or not InputDflImg.has_data():
            print('\t################ No landmarks in file {}'.format(imgFile))
            return False
    else:
        return False
    return True
    
def ispakFile(strpath:str)->bool:
    if os.path.isfile(strpath) and os.path.getsize(strpath)!=0:
        return any(strpath.endswith(extension) for extension in FACEPAK_EXTENSIONS)
    return False


def isImgFolder(strpath:str):
    if not(strpath is None) and os.path.isdir(strpath):
        endswith_tuple = tuple(IMG_EXTENSIONS)
        files=os.listdir(strpath)
        if len(files)==0:
            return False
        for fi in files:
            if fi.endswith(endswith_tuple) and isDFLAlignedImg(os.path.join(strpath,fi)):
                return True
    return False



def isAlignedFolder(strpath:str):
    """检查是否是DFL头像文件夹
    """
    if not(strpath is None) and os.path.isdir(strpath):
        ALIGNEDendswith_tuple = tuple(ALIGNED_EXTENSIONS)
        FACEPAKswith_tuple = tuple(FACEPAK_EXTENSIONS)
        #bool_endswith = path.endswith(endswith_tuple)
        files=os.listdir(strpath)
        if len(files)==0:
            return False
        index=0
        for fi in files:
            if (fi.endswith(ALIGNEDendswith_tuple) and isDFLAlignedImg(os.path.join(strpath,fi))) or fi.endswith(FACEPAKswith_tuple):
                return True
            index=index+1
            #设置一个检查次数的约束条件
            if index>=10:
                return False
    return False

def is_alignedFolder_hasContent(strpath:str):
    return isAlignedFolder(strpath)
    """
    if not os.path.exists(strpath) and not os.path.isdir(strpath):
        return False
    if os.path.exists(os.path.join(strpath,"faceset.pak")):
        return True
    for img in os.listdir(strpath):
        if img.endswith()
    imgs=glob.iglob(os.path.join(strpath,"*.jpg"))
    if imgs is None:
        imgs=glob.iglob(os.path.join(strpath,"*.JPG"))
    if imgs is None or len(imgs)==0:
        return False
    if isDFLAlignedImg(imgs[0]):
        return True
    return False
    """

VIDEO_EXTENSIONS=[
    '.avi','.AVI','.mp4','.MP4','.mkv','.MKV',
]

def isVideoFile(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)



##载入
def loadVideo(strpath:str):
    videopath=None
    thumbnail=None
    #try:
    print(strpath)
    thumbnail=util.get_randomframe_from_video(strpath)
    #except Exception:
    #    print("DFLoperate.py:loadVideo(strpath:str)载入video文件错误!")
    #    thumbnail=None
    #    return [videopath,thumbnail]
    
    videopath=strpath
    return [videopath,thumbnail]

class alignedInfoClass(object):
    name:str=None
    path:str=None
    alignedthumbnail=None
    def __init__(self,name:str,path:str,alignedthumbnail) -> None:
        super().__init__()
        self.name=name
        self.path=path
        self.alignedthumbnail=alignedthumbnail

def get_alignedFolder_info(strpath:str):
    paks=glob.glob(os.path.join(strpath,"*.pak"))
    
    if paks!=None and len(paks)!=0:        ##检查出pak文件
        if not os.path.exists(os.path.join(strpath,"alignedthumbnail.jpg")):
            return alignedInfoClass(os.path.basename(strpath),strpath,None) ##pak文件暂时无法读取预览图
        try:
            alignedThumbnail=cv2.imread(os.path.join(strpath,"alignedthumbnail.jpg"),cv2.IMREAD_COLOR)
            alignedThumbnail= cv2.cvtColor(alignedThumbnail, cv2.COLOR_BGR2RGB)
        except Exception:
            return alignedInfoClass(os.path.basename(strpath),strpath,None) ##pak文件暂时无法读取预览图
        else:
            return alignedInfoClass(os.path.basename(strpath),strpath,alignedThumbnail)

    jpgs=glob.iglob(os.path.join(strpath,"*.jpg"))
    if jpgs==None or len(jpgs)==0:
        jpgs=glob.iglob(os.path.join(strpath,"*.JPG"))
    for jpgfile in jpgs:
        if os.path.isfile(jpgfile) and isDFLAlignedImg(jpgfile):
            try:
                alignedThumbnail=cv2.imread(jpgfile,cv2.IMREAD_COLOR)
                alignedThumbnail= cv2.cvtColor(alignedThumbnail, cv2.COLOR_BGR2RGB)
            except Exception:
                print("get_alignedForlder_infor函数载入jpg图像预览错误！")
                pass
            else:
                return alignedInfoClass(os.path.basename(strpath),strpath,alignedThumbnail)
    return None

def write_alignedFolder_thumbnail(strpath:str,alignedthumbnail):
    if not os.path.exists(strpath):
        raise Exception(strpath+":未发现此文件夹！")
    if os.path.exists(os.path.join(strpath,"alignedthumbnail.jpg")):
        return False
    try:
        alignedThumbnail= cv2.cvtColor(alignedthumbnail, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(strpath,"alignedthumbnail.jpg"),alignedThumbnail)
    except Exception:
        print("write_alignedFolder_thumbnail():写入图像错误！")
        return False
    else:
        return True
    

def is_imgFolder_hasContent(strpath:str):
############################
#检查文件夹是否有图像文件
############################
    if not os.path.exists(strpath) and not os.path.isdir(strpath):
        return False
    files=os.listdir(strpath)
    #如果文件夹没内容，则返回isFolderEmpty=True
    if len(files)==0:
        return False
    for fi in files:
        if isImgFile(fi):
            return True
    return False



def get_imgFolder_info(strpath:str):
    #imgFolderPath=None
    thumbnail=None
    alignedInfos=None

    #如果异常，则返回空isFolderEmpty=True
    if not os.path.exists(strpath) or not os.path.isdir(strpath):
        return [None,None,None]
    
    files=os.listdir(strpath)
    #如果文件夹没内容，则返回isFolderEmpty=True
    if len(files)==0:
        return [None,None,None]

    #开始遍历目录
    # 如果是找到预览图，则后续文件pass，文件夹则检查是否脸图文件夹
    is_got_img=False
    for fi in files:
        if not is_got_img and os.path.isfile(os.path.join(strpath,fi)):
            if isImgFile(fi):
                try:
                    thumbnail=cv2.imread(os.path.join(strpath,fi),cv2.IMREAD_COLOR)
                    thumbnail= cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                except Exception:
                    print("get_imgFolder_info():载入"+fi+"图像预览错误！")
                    pass
                else:
                    is_got_img=True
        elif os.path.isdir(os.path.join(strpath,fi)):
            alignedinfoitem=get_alignedFolder_info(os.path.join(strpath,fi))
            if not (alignedinfoitem is None):
                if alignedInfos is None:
                    alignedInfos=[alignedInfoClass]
                alignedInfos.append(alignedinfoitem)
    
    if thumbnail is None:
        return [None,None,None]
    else:
        return [strpath,thumbnail,alignedInfos]

    """    ### 如果是.pak文件
        if fi.endswith(".pak") and os.path.isfile(os.path.join(strpath,fi)):   
            isFolderEmpty=False
            alignedPakPath=os.path.join(strpath,fi)
            return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]
        ### 如果是图像文件
        elif isImgFile(fi) and os.path.isfile(os.path.join(strpath,fi)):
            ### 如果是头像文件
            if isDFLAlignedImg(os.path.join(strpath,fi)):
                try:
                    alignedThumbnail=cv2.imread(os.path.join(strpath,fi),cv2.IMREAD_COLOR)
                    alignedThumbnail= cv2.cvtColor(alignedThumbnail, cv2.COLOR_BGR2RGB)
                except Exception:
                    print("get_typeandthumbnail_byImgorAlignedFolder函数载入jpg图像预览错误！")
                    alignedThumbnail=None
                    return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]

                alignedFolderPath=strpath
                isFolderEmpty=False     #如果没异常则处理
                return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]
            #如果不是头像文件
            else:
                try:
                    thumbnail=cv2.imread(os.path.join(strpath,fi),cv2.IMREAD_COLOR)
                    thumbnail= cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                except Exception:
                    print("get_typeandthumbnail_byImgorAlignedFolder函数载入jpg图像预览错误！")
                    thumbnail=None
                    return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]
                imgFolderPath=strpath
                isFolderEmpty=False
                alignedpath=os.path.join(strpath,"aligned")
                if os.path.exists(alignedpath) and os.path.isdir(alignedpath):
                    for fi in os.listdir(alignedpath):
                        if fi.endswith(".pak") and os.path.isfile(os.path.join(strpath),fi):   
                            isFolderEmpty=False
                            alignedPakPath=os.path.join(strpath,fi)
                            return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]
                        elif fi.endswith(".jpg") and os.path.isfile(os.path.join(strpath,fi)):
                            if isDFLAlignedImg(os.path.join(strpath,fi)):
                                try:
                                    alignedThumbnail=cv2.imread(os.path.join(strpath,fi),cv2.IMREAD_COLOR)
                                    alignedThumbnail= cv2.cvtColor(alignedThumbnail, cv2.COLOR_BGR2RGB)
                                except Exception:
                                    print("get_typeandthumbnail_byImgorAlignedFolder函数载入jpg图像预览错误！")
                                    alignedThumbnail=None
                                    alignedFolderPath=strpath
                                    return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]
                                isFolderEmpty=False
                                return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]
                            else:
                                break
                return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]
    return [isFolderEmpty,imgFolderPath,alignedFolderPath,alignedPakPath,thumbnail,alignedThumbnail]"""

def write_bytes_safe(p:Path, bytes_data):
    """
    writes to .tmp first and then rename to target filename
    """
    p_tmp = p.parent / (p.name + '.tmp')
    p_tmp.write_bytes(bytes_data)
    if p.exists():
        p.unlink()
    p_tmp.rename (p)

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry

def get_image_paths(dir_path, image_extensions=IMG_EXTENSIONS, subdirs=False, return_Path_class=False):
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():

        if subdirs:
            gen = scantree(str(dir_path))
        else:
            gen = scandir(str(dir_path))

        for x in list(gen):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append( x.path if not return_Path_class else Path(x.path) )
    return sorted(result)

def get_image_unique_filestem_paths(dir_path, verbose_print_func=None):
    result = get_image_paths(dir_path)
    result_dup = set()

    for f in result[:]:
        f_stem = Path(f).stem
        if f_stem in result_dup:
            result.remove(f)
            if verbose_print_func is not None:
                verbose_print_func ("Duplicate filenames are not allowed, skipping: %s" % Path(f).name )
            continue
        result_dup.add(f_stem)

    return sorted(result)

def get_paths(dir_path):
    dir_path = Path (dir_path)

    if dir_path.exists():
        return [ Path(x) for x in sorted([ x.path for x in list(scandir(str(dir_path))) ]) ]
    else:
        return []
        
def get_file_paths(dir_path):
    dir_path = Path (dir_path)

    if dir_path.exists():
        return [ Path(x) for x in sorted([ x.path for x in list(scandir(str(dir_path))) if x.is_file() ]) ]
    else:
        return []

def get_all_dir_names(dir_path):
    dir_path = Path (dir_path)

    if dir_path.exists():
        return sorted([ x.name for x in list(scandir(str(dir_path))) if x.is_dir() ])
    else:
        return []

def get_all_dir_names_startswith (dir_path, startswith):
    dir_path = Path (dir_path)
    startswith = startswith.lower()

    result = []
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if x.name.lower().startswith(startswith):
                result.append ( x.name[len(startswith):] )
    return sorted(result)

def get_first_file_by_stem (dir_path, stem, exts=None):
    dir_path = Path (dir_path)
    stem = stem.lower()

    if dir_path.exists():
        for x in sorted(list(scandir(str(dir_path))), key=lambda x: x.name):
            if not x.is_file():
                continue
            xp = Path(x.path)
            if xp.stem.lower() == stem and (exts is None or xp.suffix.lower() in exts):
                return xp

    return None

def move_all_files (src_dir_path, dst_dir_path):
    paths = get_file_paths(src_dir_path)
    for p in paths:
        p = Path(p)
        p.rename ( Path(dst_dir_path) / p.name )

def delete_all_files (dir_path):
    paths = get_file_paths(dir_path)
    for p in paths:
        p = Path(p)
        p.unlink()



def loadDFLWorkspace(strpath:str):
    data_dst_video=None
    data_dst_img=None
    data_dst_aligned=None
    data_dst_thumbnail=None
    data_dst_aligned_thumbnail=None #头像显示
    
    
    data_src_video=None
    data_src_img=None
    data_src_aligned=None
    data_src_thumbnail=None
    data_src_aligned_thumbnail=None #头像显示
    
    models=None

    ## 检查src视频
    try:
        if os.path.exists(os.path.join(strpath,"data_src.mp4")) and os.path.isfile(os.path.join(strpath,"data_src.mp4")):
            data_src_video=os.path.join(strpath,"data_src.mp4")
            data_src_thumbnail=util.read_frame_as_jpeg(data_src_video,0)
        elif os.path.exists(os.path.join(strpath,"data_src.avi")) and os.path.isfile(os.path.join(strpath,"data_src.avi")):
            data_src_video=os.path.join(strpath,"data_src.avi")
            data_src_thumbnail=util.read_frame_as_jpeg(data_src_video,0)
        elif os.path.exists(os.path.join(strpath,"data_src.mkv")) and os.path.isfile(os.path.join(strpath,"data_src.mkv")):
            data_src_video=os.path.join(strpath,"data_src.mkv")
            data_src_thumbnail=util.read_frame_as_jpeg(data_src_video,0)
    except Exception:
        pass
    ## 检查dst视频
    try:
        if os.path.exists(os.path.join(strpath,"data_dst.mp4")) and os.path.isfile(os.path.join(strpath,"data_dst.mp4")):
            data_dst_video=os.path.join(strpath,"data_dst.mp4")
            data_dst_thumbnail=util.read_frame_as_jpeg(data_dst_video,0)
        elif os.path.exists(os.path.join(strpath,"data_dst.avi")) and os.path.isfile(os.path.join(strpath,"data_dst.avi")):
            data_dst_video=os.path.join(strpath,"data_dst.avi")
            data_dst_thumbnail=util.read_frame_as_jpeg(data_dst_video,0)
        elif os.path.exists(os.path.join(strpath,"data_dst.mkv")) and os.path.isfile(os.path.join(strpath,"data_dst.mkv")):
            data_dst_video=os.path.join(strpath,"data_dst.mkv")
            data_dst_thumbnail=util.read_frame_as_jpeg(data_dst_video,0)
    except Exception:
        pass

    if data_src_video==None:
        data_src_img_folder=os.path.join(strpath,"data_src")
        if os.path.exists(data_src_img_folder) and os.path.isdir(data_src_img_folder):
            files=os.listdir(data_src_img_folder)
            if len(files)!=0:
                for fi in files:
                    if isImgFile(fi):
                        data_src_thumbnail=cv2.imread(os.path.join(data_src_img_folder,fi),cv2.IMREAD_COLOR)
                        break
    
    if data_dst_video==None:
        data_dst_img_folder=os.path.join(strpath,"data_dst")
        if os.path.exists(data_dst_img_folder) and os.path.isdir(data_dst_img_folder):
            files=os.listdir(data_dst_img_folder)
            if len(files)!=0:
                for fi in files:
                    if isImgFile(fi):
                        data_dst_thumbnail=cv2.imread(os.path.join(data_dst_img_folder,fi),cv2.IMREAD_COLOR)
                        break

    data_srcalignedfolder=os.path.join(strpath,"data_src","aligned")
    if os.path.exists(data_srcalignedfolder) and os.path.isdir(data_srcalignedfolder):
        files=os.listdir(data_srcalignedfolder)
        if len(files)!=0:
            for fi in files:
                if fi.endswith(".pak"):
                    data_src_aligned=os.path.join(data_srcalignedfolder,fi)
                    break
                elif fi.endswith(".jpg"):
                    if isDFLAlignedImg(os.path.join(data_srcalignedfolder,fi)):
                        data_src_aligned_thumbnail=cv2.imread(os.path.join(data_srcalignedfolder,fi),cv2.IMREAD_COLOR)
                        break
                    break

    data_dstalignedfolder=os.path.join(strpath,"data_dst","aligned")
    if os.path.exists(data_dstalignedfolder) and os.path.isdir(data_dstalignedfolder):
        files=os.listdir(data_dstalignedfolder)
        if len(files)!=0:
            for fi in files:
                if fi.endswith(".pak"):
                    data_dst_aligned=os.path.join(data_dstalignedfolder,fi)
                    break
                elif fi.endswith(".jpg"):
                    if isDFLAlignedImg(os.path.join(data_dstalignedfolder,fi)):
                        data_dst_aligned_thumbnail=cv2.imread(os.path.join(data_dstalignedfolder,fi),cv2.IMREAD_COLOR)
                        break
                    break
    model_folder_path=os.path.join(strpath,"model")
    if os.path.exists(model_folder_path) and os.path.isdir(model_folder_path):
        files=os.listdir(model_folder_path)
        if len(files)!=0:
            for fi in files:
                if fi.endswith('SAEHD_data.dat') or fi.endswith(f'AMP_data.dat') or  fi.endswith(f'Quick96_data.dat'):
                    if models==None:
                        models=[]
                    var_str=fi.split('_')[0]+"_"+fi.split('_')[1]
                    models.append(var_str)
                
    return [data_dst_video,
            data_dst_img,
            data_dst_aligned,
            data_dst_thumbnail,
            data_dst_aligned_thumbnail,
            data_src_video,
            data_src_img,
            data_src_aligned,
            data_src_thumbnail,
            data_src_aligned_thumbnail,
            models]