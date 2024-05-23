# -*- coding:utf-8 -*-

import os, sys, shutil, time
import numpy as np
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from DFLIMG import DFLIMG
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Value

func = int(sys.argv[1])
if func == 1:
    import xgboost as xgb
    modelpath = r'_internal\DeepFaceLab\ErrFaceFilter\CL.model'
    if not os.path.exists(os.path.join(os.getcwd(), modelpath)):
        raise FileNotFoundError("未发现模型文件, 请将提供的模型文件放置于源代码目录内!")
    Classifier = xgb.XGBClassifier()
    Classifier.load_model(modelpath)
elif func == 2:
    import cv2
    jaw = (255, 255, 255)
    eyelash = (255, 255, 255)
    nose = (255, 255, 255)
    eyes = (255, 255, 255)
    mouth = (255, 255, 255)

def LstCleaning(lst: list, path):
    dellst = []
    for it in lst:
        if not (it.lower().endswith(('.jpg', '.jpeg'))):
            dellst.append(it)
    for it in dellst:
        lst.remove(it)
    if len(lst) < 1:
        raise FileNotFoundError('所选目录内未发现对象,请检查后重新运行!')
    else:
        print('图像数量: \033[32m%d\033[0m' % len(lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))

def draw(lst, fpath, lmdir, cnt):
    for fname in lst:
        with cnt.get_lock():
            cnt.value += 1
        path = Path(fpath + '/' + fname)
        dflimg = DFLIMG.load(path)
        if dflimg is None or not dflimg.has_data():
            print(f'\n{fname} 不是有效的DFL头像文件!')
            continue
        else:
            x, y = np.split(dflimg.get_landmarks(), 2, axis=1)
        size = dflimg.get_shape()
        im = np.zeros(size, dtype=np.uint8)
        if size[0] <= 768:
            line_width = 5
        else:
            line_width = 12

        for i in range(0, 16):  # jaw
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), jaw, line_width)

        for i in range(17, 26):  # eyelash
            if i == 21:
                continue
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), eyelash, line_width)

        for i in range(27, 35):  # nose
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), nose, line_width)

        for i in range(36, 47):  # eyes
            if i == 41:
                img = cv2.line(im, (int(x[41][0]), int(y[41][0])), (int(x[36][0]), int(y[36][0])), eyes, line_width)
                continue
            if i == 46:
                img = cv2.line(im, (int(x[47][0]), int(y[47][0])), (int(x[42][0]), int(y[42][0])), eyes, line_width)
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), eyes, line_width)

        for i in range(48, 68):  # mouth & lips
            if i == 59:
                img = cv2.line(im, (int(x[59][0]), int(y[59][0])), (int(x[48][0]), int(y[48][0])), mouth, line_width)
                continue
            if i == 67:
                img = cv2.line(im, (int(x[67][0]), int(y[67][0])), (int(x[60][0]), int(y[60][0])), mouth, line_width)
                continue
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), mouth, line_width)

        filename = os.path.join(lmdir, fname)
        cv2.imwrite(filename, img)

def CF(flst, path, errfacelst, cnt):
    for fname in flst:
        with cnt.get_lock():
            cnt.value += 1
        dflimg = DFLIMG.load(Path(path + '/' + fname))
        if dflimg is None or not dflimg.has_data():
            print(f'\n{fname} 不是有效的DFL头像文件!')
            continue
        else:
            lm = dflimg.get_landmarks()
        h = dflimg.get_shape()[0]
        if h != 512:
            lm = lm * 512 / h
        cat = Classifier.predict(lm.reshape((1, 136)).astype(np.int64))
        if cat == 1:
            errfacelst.append(fname)

if __name__ == '__main__':
    while 1:
        os.system('cls')
        print("➜ 免 费 工 具, 放 心 白 嫖\n\033[33m➜ 作 者: 吃 果 子 的 果 子 狸\033[0m")

        path = input('请拖入存放DFL头像的文件夹(例如aligned), 并按下 Enter 键\n(\033[33m警告: 目录及头像文件名请勿出现任何中文及空格\033[0m)\n➜ ')
        if os.path.isdir(path):
            imagelst = os.listdir(path)
        else:
            raise ValueError('请输入正确的目录')
        os.system('cls')
        print('已选择目录: ', path)
        LstCleaning(imagelst, path)
        if func == 3:
            print('脚本功能: 错脸清理')
            indir = os.path.join(path, 'Landmarks')
            if not os.path.exists(indir):
                print('\n未找到Landmarks目录, 请检查后再次尝试!\n')
                input('按任意键重启脚本...')
                continue
            lmlst = os.listdir(indir)
            print('%s目录, ' % indir, end='')
            LstCleaning(lmlst, path)
            if len(lmlst) < len(imagelst):
                trash = list(set(imagelst) - set(lmlst))
            else:
                print('\n未检测到有Landmarks被剔除!\n')
                input('按任意键重启脚本...')
                continue
            movepath = os.path.join(path, 'errFace')
            if not os.path.exists(movepath):
                os.makedirs(movepath)
            print('清理数量: \033[33m%d\033[0m' % len(trash))
            for it in tqdm(trash, desc='移动进程'):
                src = os.path.join(path, it)
                dst = os.path.join(movepath, it)
                try:
                    shutil.move(src, dst)
                except:
                    print('%s 文件移动失败!', it)
            print('\n错误头像已移动到 %s\n' % movepath)
            input('按任意键重启脚本...')
        else:
            cpu = mp.cpu_count() - 1
            amount = len(imagelst) // cpu
            cpu = 1 if amount == 0 else cpu
            checklst = []
            for i in range(cpu):
                if i == cpu - 1:
                    checklst.append(imagelst[i * amount:])
                    break
                checklst.append(imagelst[i * amount:(i + 1) * amount])
            progresscnt = Value('L', 0)
            if func == 1:
                print('脚本功能: 自动筛查错脸')
                errfacelst = mp.Manager().list()
                mptask = [mp.Process(target=CF, args=(checklst[i], path, errfacelst, progresscnt, ), daemon=True) for i in range(cpu)]
                progress = tqdm(desc='检测进程', total=len(imagelst))
            elif func == 2:
                print('脚本功能: 轮廓勾画')
                LmDir = os.path.join(path, 'Landmarks')
                if not os.path.exists(LmDir):
                    os.mkdir(LmDir)
                mptask = [mp.Process(target=draw, args=(checklst[i], path, LmDir, progresscnt, ), daemon=True) for i in range(cpu)]
                progress = tqdm(desc='绘画进程', total=len(imagelst))
            [t.start() for t in mptask]
            while progress.n < len(imagelst):
                progress.n = progresscnt.value
                progress.refresh()
            progress.close()
            progress = None
            [t.join() for t in mptask]
            if func == 2:
                input('\n任务已完成, 按任意键重启脚本...\n')
                continue

            if len(errfacelst) < 1:
                print('\n目录内似乎并没有错误的脸!')
                input('\n任务已完成, 按任意键重启脚本...\n')
                continue
            errfacepath = os.path.join(path, 'errFace')
            if not os.path.exists(errfacepath):
                os.mkdir(errfacepath)
            for it in tqdm(errfacelst, desc='移动进程'):
                src = os.path.join(path, it)
                dst = os.path.join(errfacepath, it)
                try:
                    shutil.move(src, dst)
                except:
                    print('提示: %s 文件未能成功移动,请手动处理!' % it)
            print('\n找到了 \033[33m%d\033[0m 张似乎错误的头像, 并移动到了 %s' % (len(errfacelst), errfacepath))
            input('\n任务已完成, 按任意键重启脚本...\n')