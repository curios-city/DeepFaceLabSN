#%% -*- coding:utf-8 -*-
"""
作者：yangala@dfldata.xyz
日期：2021-08-31 16:51:22

"""
#=====================================================
__version__ = '0.0.1'

import sys
import os

import random

from pathlib import Path
import shutil

from DFLIMG import DFLIMG,DFLJPG
from facelib import LandmarksProcessor,FaceType

from core.interact import interact as io

def getargv():
    if sys.argv.__len__()<=1:
        return ""
    return sys.argv[1]


def check(f,cmd): # 符合条件的返回True

    dflimg = DFLJPG.load(f)

    if dflimg is None or not dflimg.has_data():
        print(f"{f.name} is not a dfl image file")
        return False,0,0

    ft = dflimg.get_face_type()
    ft = FaceType.fromString(ft)
    # print(int(ft),type(ft))
    ft = int(ft)
    # HALF = 0
    # MID_FULL = 1
    # FULL = 2
    # FULL_NO_ALIGN = 3
    # WHOLE_FACE = 4
    # HEAD = 10
    # HEAD_NO_ALIGN = 20
    # MARK_ONLY = 100


    pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(dflimg.get_landmarks(), size=dflimg.get_shape()[1])

    k=180./3.14159

    # pitch >40 仰头 <-40 看地
    # yaw >45 向左  <-45 向右

    y, x, roll = pitch*k, yaw*k, roll*k
    r = random.random()

    # print(exec(cmd,{'x':x,'y':y}),cmd,y,x,f)
    # print(eval(cmd),cmd,y,x,f)

    # return abs(x)>10 and abs(y)>10
    return eval(cmd),x,y




sort_func_methods = {
    '大角度': ("大角度：左右朝向大于40°，上下大于40° ", '(abs(x)>=40 or abs(y)>=40)'),
    '上下大角度': ("上下大角度：上下大于40° ", '(abs(y)>=40)'),
    '上下中等角度': ("上下中等角度：上下大于30° ", '(abs(y)>=30)'),
    '左右大角度': ("左右大角度：左右大于40° ", '(abs(x)>=40)'),
    '抽取20%的正脸': ("抽取20%的正脸：角度小于20°的头像随机抽取其中20% ", '( abs(x)<20 and abs(y)<20 and r<0.2 )'),
    '非wf脸': ("非wf脸：根据脸型进行筛选", '(ft != 4)'),
    '自定义': ("自定义：x表示左右，y表示上下,r表示0-1的随机值,ft表示脸型 ", ''),
}

if __name__ == '__main__':

    print('yaw_image_filter.py 启动......')
    a=getargv()

    if os.path.exists(a):
        pass
    else:
        # print('不存在jpg所在目录：', a)
        a=input('路径无效，请输入jpg所在目录：')

    # jpg所在目录
    alignedpath=Path(a)
    if alignedpath.is_file():
        alignedpath = alignedpath.parent


    print(f"\r\n头像所在目录为:{alignedpath}\r\n")

    # 菜单
    key_list = list(sort_func_methods.keys())
    for i, key in enumerate(key_list):
        desc, func = sort_func_methods[key]
        io.log_info(f"[{i}] {desc+' -> '+func}")

    io.log_info("")
    id = io.input_int("", 0, valid_list=[*range(len(key_list))] )

    sort_by_method = key_list[id]

    cmd = sort_func_methods[sort_by_method][1]
    print(sort_by_method,cmd)

    if cmd == '':
        print('''
    x 表示左右，大于0表示朝左，小于0表示朝右
    y 表示上下，大于0表示抬头，小于0表示低头
    r 表示0-1的随机值
    
    ft 脸型 wf=4 f=2 head=10
    
    abs()取绝对值的函数
    
    and 并且
    or 或者
    not 非
    
    示例： 
    r<0.2           表示随机抽取20%的头像
    x>20 and y<-20  表示抽取朝左20°并且低头20°的头像 
    ft!=4           表示抽取非wf的脸型
        ''')
        cmd = input('请输入判断依据：')
    # if cmd == '':
    #     print('cmd为空，退出')
    #     exit(0)

    # print(sort_by_method, cmd)


    # 目标目录
    mubiao = 'aligned_'+sort_by_method
    a = input(f'请输入目标目录名，直接按回车则默认为 {mubiao} :')
    if a == '':
        a = mubiao
    filterpath = alignedpath.parent / (a)
    if not filterpath.exists():
        filterpath.mkdir(parents=True)

    # 拷贝还是移动
    a = input(f'是拷贝文件还是移动文件？1拷贝，2移动，直接按回车则默认为 1拷贝 :')
    if a=='' or a=='1':
        cpimg = '拷贝：'
    else:
        cpimg = '移动：'

    #
    #
    #
    # cmd = '(abs(x)>=40 or abs(y)>=30) and r<0.1 '
    # # cmd = 'x>10.0'

    # 统计角度分布
    tongji = [ [ 0 for b in range(7)   ] for a in range(7) ]
    def xyidx(x):
        return min(6,max(0,int(x+90+14.99999)//30))

    cnt = 0
    cntcopy = 0
    for f in alignedpath.glob('*.*'):
        # print(f.name)
        if f.is_file():
            cnt += 1
            rst,x,y = check(f,cmd)
            tongji[xyidx(y)][xyidx(x)] += 1
            if rst:
                # print(f.name)
                print(cnt,cpimg,f.name)
                dst = filterpath / f.name
                if cpimg == '拷贝：':
                    shutil.copy(f,dst)
                else:
                    shutil.move(f,dst)
                cntcopy += 1
            else:
                print(cnt,f.name)

    print()
    print(f"处理结果:{filterpath}")
    print(f'共有文件 {cnt} 个，{cpimg} {cntcopy} 个 {sort_by_method}')

    jiaodu = ['','<-75','-60 ','-30 ','0±15','30  ','60  ','>75 ']
    print()
    print(' '+'\t'.join(jiaodu))
    print('------------------------------------------------------------')
    for a in range(7):
        print(jiaodu[a+1] +'\t| '+ '\t| '.join( [str(b if b>0 else '  ') for b in tongji[a]]  ))
        print('------------------------------------------------------------')



