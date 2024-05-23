#-*- coding: utf-8 -*-

#from genericpath import exists, isfile
import os
from PyQt5.QtWidgets import *
import ffmpeg
import cv2
import numpy
from PyQt5.QtGui import *
from numpy.lib.function_base import copy
from UI.utils import hardware
from UI.utils.DFLIMG.DFLJPG import DFLJPG
from enum import Enum
import copy
from PyQt5 import QtCore


class statusEnum(Enum):
        loading_start=0
        loading=1
        stoping_start=2
        stoping_end=3
        

def read_frame_as_jpeg(in_file:str, frame_num:int):
    """返回RGB图像
    """
    try:
        out, err = (
            ffmpeg.input(in_file)
                .filter('select', 'gte(n,{})'.format(frame_num))
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True)
        )
    except Exception as err:
        print(err.__str__())
        raise
    try:
        image_array = numpy.asarray(bytearray(out), dtype="uint8")
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        #cv2.imshow('frame', image)
        #cv2.waitKey()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as ex:
        print(ex.__str__())
        raise

def read_frame_by_time(in_file:str, time):
    try:
        out, err = (
            ffmpeg.input(in_file, ss=time)
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True)
        )
    except ffmpeg.Error as err:
        print(err.__str__())
        raise
    image_array = numpy.asarray(bytearray(out), dtype="uint8")
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        #cv2.imshow('frame', image)
        #cv2.waitKey()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


        
def get_duration_from_video(videopath:str):
    try:
        ##调用ffmpeg处理视频
        probe = ffmpeg.probe(videopath)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            return None
        duration=int(float(video_stream['duration']))
        return duration
    except ffmpeg.Error as err:
        return None


def duration_to_Qtime(duration:int):
    hh=int(duration/60/60)
    mm=int(duration/60)-hh*60
    ss=duration-hh*60*60-mm*60
    return QtCore.QTime(hh,mm,ss)

def vidreadmeta(fn):
    if not os.path.exists(fn):
        raise FileNotFoundError
    probe = ffmpeg.probe(fn)
    for stream in probe['streams']:
        if stream['codec_type'] == 'video':
            meta = {
                'width': int(stream['width']),
                'height': int(stream['height']),
                'duration': float(probe['format']['duration'])
            }
            return meta
    return None 

def get_randomframe_from_video(videopath:str):
    duration=get_duration_from_video(videopath)
    import random
    random_time = random.uniform(0,float(duration))
    return read_frame_by_time(videopath,random_time)

"""#--------------------------------------------
#                   选择视频后的界面显示
#---------------------------------------------
def showVideo(var_path:str,VideographicsView:QGraphicsView,QSlider_VideoHorizontal:QSlider,
                QLineEdit_startTime:QLineEdit,
                QLineEdit_endTime:QLineEdit,
                QLineEdit_FPS:QLineEdit,
                QComboBox_audioTracks:QComboBox,
                QCheckbox_videoCute:QCheckBox)->str:
    try:
        if not os.path.exists(var_path):
            setEnableVideoWidgets(False,VideographicsView,QSlider_VideoHorizontal,QLineEdit_startTime,
                                    QLineEdit_endTime,QLineEdit_FPS,QComboBox_audioTracks,QCheckbox_videoCute)
            return "您选择的视频路径不存在！"

        if not os.path.isfile(var_path):
            setEnableVideoWidgets(False,VideographicsView,QSlider_VideoHorizontal,QLineEdit_startTime,
                                    QLineEdit_endTime,QLineEdit_FPS,QComboBox_audioTracks,QCheckbox_videoCute)
            return "您选择视频路径不是一个文件！"
                
        ##调用ffmpeg处理视频
        probe = ffmpeg.probe(var_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            setEnableVideoWidgets(False,VideographicsView,QSlider_VideoHorizontal,QLineEdit_startTime,
                                    QLineEdit_endTime,QLineEdit_FPS,QComboBox_audioTracks,QCheckbox_videoCute)
            return "该文件无法正常读取视频！"

        ##video预览图片显示
        view=read_frame_as_jpeg(var_path,1)
        image_array = numpy.asarray(bytearray(view), dtype="uint8")
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        #cv2.imshow('frame', image)
        #cv2.waitKey()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.shape[1]                                        #获取图像大小
        y = img.shape[0]
        zoomscale=min(VideographicsView.width()/x,
                        VideographicsView.height()/y)                                        #图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        var_pixmapitem=QGraphicsPixmapItem(pix)                      #创建像素图元
        var_pixmapitem.setScale(zoomscale) 
        #self.item.setScale(self.zoomscale)
        scene=QGraphicsScene()                          #创建场景
        scene.addItem(var_pixmapitem)
        VideographicsView.setScene(scene) 

        ##滑动条定义
        total_frames = int(video_stream['nb_frames']) 
        QSlider_VideoHorizontal.setMinimum(1)
        QSlider_VideoHorizontal.setMaximum(total_frames)
        
        ##起始时间显示
        duration=video_stream['duration']
        QLineEdit_startTime.setText("0")
        QLineEdit_startTime.setValidator(QtGui.QIntValidator(0,int(float(duration))))
        QLineEdit_endTime.setText(duration)
        QLineEdit_endTime.setValidator(QtGui.QIntValidator(1,int(float(duration))))
        
        #显示fps
        QLineEdit_FPS.setText(str(int(video_stream['r_frame_rate'].split('/')[0])/int(video_stream['r_frame_rate'].split('/')[1])))
        
        #显示音轨
        list_audiosindexs=[]
        for stream in probe['streams']:
            if stream['codec_type'] == 'audio':
                list_audiosindexs.append(str(stream['index']))
        QComboBox_audioTracks.clear()
        if list_audiosindexs:
            QComboBox_audioTracks.addItems(list_audiosindexs)

        #width = int(video_stream['width'])
        #height = int(video_stream['height'])
        setEnableVideoWidgets(True,VideographicsView,QSlider_VideoHorizontal,QLineEdit_startTime,
                                    QLineEdit_endTime,QLineEdit_FPS,QComboBox_audioTracks,QCheckbox_videoCute)
        return ""

    except ffmpeg.Error as err:
        setEnableVideoWidgets(False,VideographicsView,QSlider_VideoHorizontal,QLineEdit_startTime,
                                    QLineEdit_endTime,QLineEdit_FPS,QComboBox_audioTracks,QCheckbox_videoCute)
        return str(err.stderr, encoding='utf8')
"""
def showImg(var_path:str,ImgView:QGraphicsItem):
    img=cv2.imread(var_path)                              #读取图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              #转换图像通道
    x = img.shape[1]                                        #获取图像大小
    y = img.shape[0]
    zoomscale=1                                        #图片放缩尺度
    frame = QImage(img, x, y, QImage.Format_RGB888)
    pix = QPixmap.fromImage(frame)
    ImgView=QGraphicsPixmapItem(pix)                      #创建像素图元

def get_safethumbtail(thumbtail):
    if thumbtail is None:
        # 使用Numpy创建一张(1024×1024)纸
        img = numpy.zeros((1024,1024,3), numpy.uint8)
        # 使用白色填充图片区域,默认为黑色
        img.fill(255)
        return img
    else:
        return thumbtail

def convert_img_to_Qicon(img):
    x = img.shape[1]                                        #获取图像大小
    y = img.shape[0]
    #zoomscale=1                                        #图片放缩尺度
    frame = QImage(img, x, y, QImage.Format_RGB888)
    pix = QPixmap.fromImage(frame)
    ico=QIcon(pix)
    return ico

def convert_img_to_QGraphicsPixmapItem(img,ImgView:QGraphicsItem):
    x = img.shape[1]                                        #获取图像大小
    y = img.shape[0]
    #zoomscale=1                                        #图片放缩尺度
    frame = QImage(img, x, y, QImage.Format_RGB888)
    pix = QPixmap.fromImage(frame)
    ImgView=QGraphicsPixmapItem(pix)                      #创建像素图元


def getModelList(var_ModelWorkSpacePath:str,var_defaultModelFolderName:str)->list:
    var_Modelpath=os.path.join(var_ModelWorkSpacePath,var_defaultModelFolderName)
    if (not os.path.exists(var_Modelpath)) or (not os.path.isdir(var_Modelpath)):
        os.mkdir(var_Modelpath)
        return None
        
    list_modelNames=[]
    for file in os.listdir(var_Modelpath):
        if os.path.splitext(file)[1]==".npy":
            list_tmp=str.split("_",3)
            if list_modelNames:
                for modelname in list_modelNames:
                    if modelname==str(list_tmp[0]+list_tmp[1]):
                        continue
                    else:
                        list_modelNames.append(list_tmp[0]+list_tmp[1])
                        break
            else:
                list_modelNames.append(list_tmp[0]+list_tmp[1])

def str_to_bool(str):
    return True if str.lower() == 'true' else False

def strip_by_filepath(s:str):
    word = ''
    for i in s:
        if ord(i)>31 and ord(i)!=47 and ord(i)!=92 and ord(i)!=58 and ord(i)!=42 and ord(i)!=34 and ord(i)!=60 and ord(i)!=62 and ord(i)!=124 and ord(i)!=63:
            word += i            
    return word





def searchFile(fileList,filename:str)->str:
    for fi in fileList:
        if filename==fi:
            return True
    return False

def img_to_str(img):
    img_encode=cv2.imencode('.jpg',img)[1]
    data_encode=numpy.array(img_encode)
    str_encode=data_encode.tostring()
    return str_encode

def str_to_img(str_encode):
    nparr=numpy.fromstring(str_encode,numpy.uint8)
    img_decode=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return img_decode

def img_to_QImage_scale(rgb_image,resize_height:int, resize_width:int,normalization=False):
 
    #rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
        # show_image(filename,rgb_image)
        # rgb_image=Image.open(filename)
    if resize_height > 0 and resize_width > 0:
        scale=min(float(resize_width/rgb_image.shape[1]),float(resize_height/rgb_image.shape[0]))
        #rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
        #rgb_image=cv2.resize(rgb_image,(round(rgb_image.shape[1]*scale), round(rgb_image.shape[0]*scale)))
        rgb_image=cv2.resize(rgb_image,dsize=None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
    rgb_image = numpy.asanyarray(rgb_image)
    if normalization:
            # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
        # show_image("src resize image",image)
    label_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)#转化为QImage
    return label_image  

def img_to_QImage(rgb_image,normalization=False):
    #rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
        # show_image(filename,rgb_image)
        # rgb_image=Image.open(filename)
    #if resize_height > 0 and resize_width > 0:
        #rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = numpy.asanyarray(rgb_image)
    if normalization:
            # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
        # show_image("src resize image",image)
    label_image = QImage(rgb_image.data.tobytes(), rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)#转化为QImage
    return label_image  


def get_paths_from_event(event:QtCore.QEvent):
    """从event中分离路径列表
        \n返回： 错误 None
        \n正确： 路径列表
    """
    if event.mimeData().hasUrls():   # if file or link is dropped
        urls = event.mimeData().urls()
        if urls == "" or len(urls)==0:
            print("urls是空的！")
            return None
        paths=[]
        for urlitem in urls:
            strpath=urlitem.toLocalFile()
            paths.append(strpath)
        return paths
    return None