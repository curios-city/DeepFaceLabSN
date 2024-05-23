# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QFrame, QPushButton
import math
from PyQt5.QtCore import QObject, QPoint, pyqtSignal
from PyQt5.QtCore import Qt
import os,sys
from pathlib import Path

from UI.utils.DFLIMG.DFLJPG import DFLIMG
from UI.utils.DFLoperate import get_image_paths
import cv2
import numpy as np
from UI.utils.DFLoperate import get_all_dir_names,get_image_paths
#from UI.utils.DFLIMG.Sample import load_face_samples


from UI.utils.util import get_safethumbtail, img_to_QImage
from UI.utils.util import convert_img_to_Qicon
import pickle
import struct

DFLlocationRealtime=os.path.realpath("..\\")
DFLpath=os.path.join(DFLlocationRealtime,"DeepFaceLab")
sys.path.append(DFLpath)
import samplelib.SampleLoader



landmarks_68_3D = np.array( [
[-73.393523  , -29.801432   , 47.667532   ], #00
[-72.775014  , -10.949766   , 45.909403   ], #01
[-70.533638  , 7.929818     , 44.842580   ], #02
[-66.850058  , 26.074280    , 43.141114   ], #03
[-59.790187  , 42.564390    , 38.635298   ], #04
[-48.368973  , 56.481080    , 30.750622   ], #05
[-34.121101  , 67.246992    , 18.456453   ], #06
[-17.875411  , 75.056892    , 3.609035    ], #07
[0.098749    , 77.061286    , -0.881698   ], #08
[17.477031   , 74.758448    , 5.181201    ], #09
[32.648966   , 66.929021    , 19.176563   ], #10
[46.372358   , 56.311389    , 30.770570   ], #11
[57.343480   , 42.419126    , 37.628629   ], #12
[64.388482   , 25.455880    , 40.886309   ], #13
[68.212038   , 6.990805     , 42.281449   ], #14
[70.486405   , -11.666193   , 44.142567   ], #15
[71.375822   , -30.365191   , 47.140426   ], #16
[-61.119406  , -49.361602   , 14.254422   ], #17
[-51.287588  , -58.769795   , 7.268147    ], #18
[-37.804800  , -61.996155   , 0.442051    ], #19
[-24.022754  , -61.033399   , -6.606501   ], #20
[-11.635713  , -56.686759   , -11.967398  ], #21
[12.056636   , -57.391033   , -12.051204  ], #22
[25.106256   , -61.902186   , -7.315098   ], #23
[38.338588   , -62.777713   , -1.022953   ], #24
[51.191007   , -59.302347   , 5.349435    ], #25
[60.053851   , -50.190255   , 11.615746   ], #26
[0.653940    , -42.193790   , -13.380835  ], #27
[0.804809    , -30.993721   , -21.150853  ], #28
[0.992204    , -19.944596   , -29.284036  ], #29
[1.226783    , -8.414541    , -36.948060  ], #00
[-14.772472  , 2.598255     , -20.132003  ], #01
[-7.180239   , 4.751589     , -23.536684  ], #02
[0.555920    , 6.562900     , -25.944448  ], #03
[8.272499    , 4.661005     , -23.695741  ], #04
[15.214351   , 2.643046     , -20.858157  ], #05
[-46.047290  , -37.471411   , 7.037989    ], #06
[-37.674688  , -42.730510   , 3.021217    ], #07
[-27.883856  , -42.711517   , 1.353629    ], #08
[-19.648268  , -36.754742   , -0.111088   ], #09
[-28.272965  , -35.134493   , -0.147273   ], #10
[-38.082418  , -34.919043   , 1.476612    ], #11
[19.265868   , -37.032306   , -0.665746   ], #12
[27.894191   , -43.342445   , 0.247660    ], #13
[37.437529   , -43.110822   , 1.696435    ], #14
[45.170805   , -38.086515   , 4.894163    ], #15
[38.196454   , -35.532024   , 0.282961    ], #16
[28.764989   , -35.484289   , -1.172675   ], #17
[-28.916267  , 28.612716    , -2.240310   ], #18
[-17.533194  , 22.172187    , -15.934335  ], #19
[-6.684590   , 19.029051    , -22.611355  ], #20
[0.381001    , 20.721118    , -23.748437  ], #21
[8.375443    , 19.035460    , -22.721995  ], #22
[18.876618   , 22.394109    , -15.610679  ], #23
[28.794412   , 28.079924    , -3.217393   ], #24
[19.057574   , 36.298248    , -14.987997  ], #25
[8.956375    , 39.634575    , -22.554245  ], #26
[0.381549    , 40.395647    , -23.591626  ], #27
[-7.428895   , 39.836405    , -22.406106  ], #28
[-18.160634  , 36.677899    , -15.121907  ], #29
[-24.377490  , 28.677771    , -4.785684   ], #30
[-6.897633   , 25.475976    , -20.893742  ], #31
[0.340663    , 26.014269    , -22.220479  ], #32
[8.444722    , 25.326198    , -21.025520  ], #33
[24.474473   , 28.323008    , -5.712776   ], #34
[8.449166    , 30.596216    , -20.671489  ], #35
[0.205322    , 31.408738    , -21.903670  ], #36 
[-7.198266   , 30.844876    , -20.328022  ]  #37
], dtype=np.float32)

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def estimate_pitch_yaw_roll(aligned_landmarks, size=256):
    """
    returns pitch,yaw,roll [-pi/2...+pi/2]
    """
    shape = (size,size)
    focal_length = shape[1]
    camera_center = (shape[1] / 2, shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype=np.float32)

    (_, rotation_vector, _) = cv2.solvePnP(
        np.concatenate( (landmarks_68_3D[:27],   landmarks_68_3D[30:36]) , axis=0) ,
        np.concatenate( (aligned_landmarks[:27], aligned_landmarks[30:36]) , axis=0).astype(np.float32),
        camera_matrix,
        np.zeros((4, 1)) )

    pitch, yaw, roll =rotationMatrixToEulerAngles( cv2.Rodrigues(rotation_vector)[0] )
   
    half_pi = math.pi / 2.0
    pitch = np.clip ( pitch, -half_pi, half_pi )
    yaw   = np.clip ( yaw ,  -half_pi, half_pi )
    roll  = np.clip ( roll,  -half_pi, half_pi )

    return -pitch, yaw, roll

class tickclass(QObject):
    def __init__(self) -> None:
        QObject.__init__(self)
        self.pitchlist=[]
        for i,pitchtick in enumerate(pitchticks):
            if math.isinf(pitchtick) and i==0:
                minpitch=float(pitchticks[1])-2.5
                maxpitch=pitchtick
            elif math.isinf(pitchtick) and i!=0:
                minpitch=pitchtick
                maxpitch=float(pitchticks[i-1])+2.5
            else:
                minpitch=float(pitchtick)-2.5
                maxpitch=float(pitchtick)+2.5
            yawlist=[]
            for j,yawtick in enumerate(yawticks):
                if math.isinf(yawtick) and j==0:
                    minyaw=yawtick
                    maxyaw=float(yawticks[1])+2.5
                elif math.isinf(yawtick) and j!=0:
                    minyaw=float(yawticks[j-1])-2.5
                    maxyaw=yawtick
                else:
                    minyaw=float(yawtick)-2.5
                    maxyaw=float(yawtick)+2.5
                srcfilepathlist=[]
                srccout=0
                dstfilepathlist=[]
                dstcout=0
                item=[pitchtick,minpitch,maxpitch,yawtick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]
                yawlist.append(item)
            self.pitchlist.append(yawlist)
        #moveToThread必须放后面，否则信号连接不正常
        self.mainThread = QtCore.QThread()
        self.moveToThread(self.mainThread)
        #self.mainThread.started.connect(self._qthread_drop_paths_from_startpage)
        #self.mainThread.finished.connect(self.slot_mainThread_finished)
        self.mainThread.started.connect(self.load_AlignedThread)

    def getFullInfo(self,x,y):
        #print(str(x)+"."+str(y))
        yawlist=self.pitchlist[y]
        item=yawlist[x]
        #[pitchtick,minpitch,maxpitch,yawtick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]=item
        return item
        
    def getDSTfilelist(self,x,y):
        #print("p长度"+str(len(self.pitchlist)))
        print(str(x)+"."+str(y))
        yawlist=self.pitchlist[y]
        item=yawlist[x]
        [pitchtick,minpitch,maxpitch,yawtick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]=item
        return dstfilepathlist
        
    sig_statusend=pyqtSignal(bool)      #true,表示pak，false表示load
    def slot_mainThread_finished(self,b_pak:bool):
        print("finished")
        self.sig_statusend.emit(b_pak)
        

    sig_alignedpreview=pyqtSignal(bool,QPixmap)
    sig_AlignedLoadProcess=pyqtSignal(bool,int,int)

    def startload(self,srcalignedfolder:str,b_src:bool):
        print("startloaded")
        self.srcalignedfolder=srcalignedfolder
        self.b_src=b_src
        if self.mainThread.isRunning() or not self.mainThread.isFinished():
            self.mainThread.quit()
            print("线程处于执行状态！")
        try:
            self.mainThread.started.disconnect()
            self.mainThread.started.connect(self.load_AlignedThread)
            self.mainThread.start()
        except Exception as ex:
            print(ex.__str__())

    
    sig_statusstarted=pyqtSignal()
    def load_AlignedThread(self):
        srcalignedfolder=self.srcalignedfolder
        b_src=self.b_src
        self.sig_statusstarted.emit()
        paths=get_image_paths(srcalignedfolder)
        count=len(paths)-1
        b_previewed=False
        ##清空信息状态
        for ylist in self.pitchlist:
            for yitem in ylist:
                [pitchtick,minpitch,maxpitch,yawtick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]=yitem
                if b_src:
                    srcfilepathlist.clear()
                else:
                    dstfilepathlist.clear()
        for i,filepath in enumerate(paths) :
            self.sig_AlignedLoadProcess.emit(b_src,count,i)
            #print(str(i)+":start")
            try:
                filepath = Path(filepath)
                dflimg = DFLIMG.load (filepath)
                if dflimg is None or not dflimg.has_data():
                    continue
                pitch, yaw, roll = estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )
            except:
                continue

            if not b_previewed:
                thumbnail=None
                try:
                    thumbnail=cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
                    thumbnail= cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                except Exception as ex:
                    print(ex.__str__())
                img=get_safethumbtail(thumbnail)
                #img1=cv2.resize(img,(200,22),interpolation=cv2.INTER_CUBIC)
                qimg=img_to_QImage(img)
                pix=QPixmap(qimg)
                b_previewed=True

                self.sig_alignedpreview.emit(b_src,pix)
                

            #print(str(i)+":processing")
            pAngle=pitch*180/(math.pi)
            yAngle=yaw*(-180)/(math.pi)
            #print("p角度："+str(pAngle)+"  y角度："+str(yAngle))
            for yamlist in self.pitchlist:
                    [pitchtick,minpitch,maxpitch,yawtick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]=yamlist[0]
                    if pAngle <minpitch or pAngle>maxpitch:
                        continue
                    for item in yamlist:
                        [pitchtick,minpitch,maxpitch,yawtick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]=item
                        if yAngle>minyaw and yAngle<maxyaw:
                            if b_src:
                                srcfilepathlist.append(filepath)
                                #srccout+=1
                                #print("append")
                            else:
                                dstfilepathlist.append(filepath)
                                #dstcout+=1
                                #print("append")
                            #item=[pitchtick,minpitch,maxpitch,yawtick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]
            #print(str(i)+":end"+str(pitch)+":"+str(yaw))

        self.slot_mainThread_finished(b_pak=False)

    VERSION = 1
    packed_faceset_filename = 'faceset.pak'

    def startpak(self,dstpakpathstr:str,srcfolder:str,dit:dict):
        self.dstpakpathstr=dstpakpathstr
        self.srcfolder=srcfolder
        self.dit=dit
        if self.mainThread.isRunning() or not self.mainThread.isFinished():
            self.mainThread.quit()
            print("线程处于执行状态！")
        try:
            self.mainThread.started.disconnect()
            self.mainThread.started.connect(self.pakfile)
            self.mainThread.start()
        except Exception as ex:
            print(ex.__str__())


    sig_pakProcess=pyqtSignal(int,int)
    def pakfile(self):
        self.sig_statusstarted.emit()
        dstpakpathstr=self.dstpakpathstr
        srcfolder=self.srcfolder
        dit=self.dit
        image_paths = []
        mainPath=str(Path(srcfolder))
        mainPersonName=Path(srcfolder).name
        extPath=""
        if mainPersonName!="ext":
            extPersonName="ext"
        else:
            extPersonName="extent"
        b_getextPath=False
        count=28*28*2
        index=0
        for key in dit:
            strlist=dit[key]
            if not (strlist is None) and len(strlist)>0:
                for filepath in strlist:
                    if not b_getextPath:
                        extPath=str(Path(filepath).parent)
                        b_getextPath=True
                    image_paths.append(filepath)
            index+=1
            self.sig_pakProcess.emit(count,index)
        image_paths += get_image_paths(Path(srcfolder))
        

        samples = samplelib.SampleLoader.load_face_samples(image_paths)
        samples_len = len(samples)

        
        count=samples_len*2
        as_person_faceset=True

        samples_configs = []
        index=0
        for sample in samples:
            index+=1
            self.sig_pakProcess.emit(count,index)
            sample_filepath = Path(sample.filename)
            sample.filename = sample_filepath.name

            if as_person_faceset:
                if str(sample_filepath.parent)==mainPath:
                    sample.person_name = mainPersonName
                else:
                    sample.person_name = extPersonName
            samples_configs.append ( sample.get_config() )
        samples_bytes = pickle.dumps(samples_configs, 4)

        of = open(dstpakpathstr, "wb")
        of.write ( struct.pack ("Q", self.VERSION ) )
        of.write ( struct.pack ("Q", len(samples_bytes) ) )
        of.write ( samples_bytes )

        del samples_bytes   #just free mem
        del samples_configs

        sample_data_table_offset = of.tell()
        of.write ( bytes( 8*(samples_len+1) ) ) #sample data offset table

        data_start_offset = of.tell()
        offsets = []

        for sample in samples:
            index+=1
            self.sig_pakProcess.emit(count,index)
            try:
                if sample.person_name is not None:
                    if sample.person_name==mainPersonName:
                        sample_path = Path(mainPath) / sample.filename
                    else:
                        sample_path = Path(extPath) / sample.filename

                else:
                    sample_path = Path(mainPath) / sample.filename


                with open(sample_path, "rb") as f:
                   b = f.read()


                offsets.append ( of.tell() - data_start_offset )
                of.write(b)
            except:
                print(f"error while processing sample {sample_path}")

        offsets.append ( of.tell() )

        of.seek(sample_data_table_offset, 0)
        for offset in offsets:
            of.write ( struct.pack("Q", offset) )
        of.seek(0,2)
        of.close()
        self.slot_mainThread_finished(b_pak=True)


pitchticks=[float('inf'),50,45,40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40,-45,float('-inf') ]
yawticks=[float('-inf'),-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,float('inf')]


class facegridframe(QFrame):
    def __init__(self,parent):
        super().__init__(parent)
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setContentsMargins(1, 1, 1, 1)
        self.gridLayout.setObjectName("gridLayout")

        self.buttonsdict={}
        for i,ptick in enumerate(pitchticks):       #注意，这是是Y轴
            if i==len(pitchticks)-1:
                self.gridLayout.setRowStretch(i,1)
                self.gridLayout.setRowStretch(i+1,0)
            else:
                self.gridLayout.setRowStretch(i,1)
            for j,ytick in enumerate(yawticks):     #注意，这是x轴
                self.gridLayout.setColumnStretch(j,1)
                if math.isinf(ptick) and i==0:
                        plabelstr="inf"
                        plabeltext='>'
                elif math.isinf(ptick) and i!=0:
                        plabelstr="uinf"
                        plabeltext='<'
                elif ptick>=0:
                        plabelstr=plabeltext=str(ptick)
                else:
                    plabelstr="u"+str(0-ptick)
                    plabeltext=str(ptick)

                if math.isinf(ytick) and j==0:
                        ylabelstr="uinf"
                        ylabeltext='<'
                elif math.isinf(ytick) and j!=0:
                        ylabelstr="inf"
                        ylabeltext='>'
                elif ytick>=0:
                        ylabeltext=ylabelstr=str(ytick)
                else:
                    ylabelstr="u"+str(0-ytick)
                    ylabeltext=str(ytick)

                #添加y轴标签
                if j==0:
                    plabel=QtWidgets.QLabel(self)
                    
                    plabel.setObjectName(u'plabel_'+plabelstr)
                    plabel.setText(plabeltext)
                    plabel.setStyleSheet("color: rgb(133,133,133);")
                    plabel.setAlignment(Qt.AlignmentFlag.AlignRight)
                    self.gridLayout.addWidget(plabel,i,j)

                #添加网格button
                btn=QtWidgets.QPushButton(self)
                keyname=u'button_'+ylabelstr+'_'+plabelstr
                btn.setObjectName(keyname)
                btn.setCheckable(False)
                #btn.setCheckable(True)
                #btn.setEnabled(False)
                #btn.setChecked(False)
                btn.setToolTip("主：0\n辅：0")
                StyleSheetstr="QPushButton{"+"border-radius:2px;\n"+"border:1px solid rgb(133,133,133);\n"+"background: rgb(0,0,0);\n"+"}\nQPushButton:checked{"+"background: yellow}"
                btn.setStyleSheet(StyleSheetstr)
                btn.setContextMenuPolicy(Qt.CustomContextMenu)
                btn.toggled.connect(self.slot_btn_toggled )
                btn.customContextMenuRequested.connect(self.slot_btn_rightClicked)
                
                self.gridLayout.addWidget(btn,i,j+1)        #第一列是标签，所以j要+1
                self.buttonsdict[keyname]=[j,i]
                

                #添加x轴标签
                if i==len(pitchticks)-1:
                    ylabel=QtWidgets.QLabel(self)
                    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                    ylabel.setSizePolicy(sizePolicy)
                    ylabel.setMaximumSize(QtCore.QSize(20, 15))
                    ylabel.setObjectName(u'ylabel_'+ylabelstr)
                    ylabel.setText(ylabeltext)
                    ylabel.setStyleSheet("color: rgb(133,133,133);")
                    ylabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.gridLayout.addWidget(ylabel,i+1,j+1)

    gridunchecked=pyqtSignal(int,int)     #x,y轴
    gridchecked=pyqtSignal(int,int)       #x,y轴
    gridrightClicked=pyqtSignal(int,int)       #x,y轴

    def slot_btn_rightClicked(self,pos):
        keyname=self.sender().objectName()
        [x,y]=self.buttonsdict[keyname]
        self.gridrightClicked.emit(x,y)

    def slot_btn_toggled(self):
        keyname=self.sender().objectName()
        [x,y]=self.buttonsdict[keyname]
        if self.sender().isChecked():
            self.gridchecked.emit(x,y)
        else:
            self.gridunchecked.emit(x,y)

    def loadfacesinfo(self,facesinfo:tickclass):
        for i,ylist in enumerate(facesinfo.pitchlist):       #注意，这是是Y轴
            for j,item in enumerate(ylist):
                [ptick,minpitch,maxpitch,ytick,minyaw,maxyaw,srcfilepathlist,srccout,dstfilepathlist,dstcout]=item
                if math.isinf(ptick) and i==0:
                        plabelstr="inf"
                elif math.isinf(ptick) and i!=0:
                        plabelstr="uinf"
                elif ptick>=0:
                        plabelstr=str(ptick)
                else:
                    plabelstr="u"+str(0-ptick)

                if math.isinf(ytick) and j==0:
                        ylabelstr="uinf"
                elif math.isinf(ytick) and j!=0:
                        ylabelstr="inf"
                elif ytick>=0:
                        ylabelstr=str(ytick)
                else:
                    ylabelstr="u"+str(0-ytick)

                btn=self.findChild(QtWidgets.QPushButton, u'button_'+ylabelstr+'_'+plabelstr)
                normalstylestr="border-radius:2px;\n"
                btn.setCheckable(False)
                btn.setChecked(False)
                if len(dstfilepathlist)==0:
                    #btn.setEnabled(False)
                    normalstylestr+="border:1px solid rgb(133,133,133);\n"
                else:
                    btn.setCheckable(True)
                    #btn.setEnabled(True)
                    normalstylestr+="border:2px solid green;\n"

                if len(srcfilepathlist)==0:
                    normalstylestr+="background: rgb(0,0,0);\n"    #背景黑色
                elif len(srcfilepathlist)<5:
                    normalstylestr+="background: rgb(50,50,50);\n" #背景白色
                elif len(srcfilepathlist)<20:
                    normalstylestr+="background: rgb(130,130,130);\n" #背景白色
                else:
                    normalstylestr+="background: rgb(230,230,230);\n" #背景白色

                StyleSheetstr="QPushButton{"+normalstylestr+"}\nQPushButton:checked{"+"background: yellow}"
                btn.setStyleSheet(StyleSheetstr)
                btn.setToolTip("主："+str(len(srcfilepathlist))+"\n辅："+str(len(dstfilepathlist)))


