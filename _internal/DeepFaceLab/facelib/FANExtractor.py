import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

from facelib import FaceType, LandmarksProcessor
from core.leras import nn

"""
ported from https://github.com/1adrianb/face-alignment
"""
class FANExtractor(object):
    def __init__ (self, landmarks_3D=False, place_model_on_cpu=False):
        
        model_path = Path(__file__).parent / ( "2DFAN.npy" if not landmarks_3D else "3DFAN.npy")
        if not model_path.exists():
            raise Exception("Unable to load FANExtractor model")

        nn.initialize(data_format="NHWC")
        tf = nn.tf

        class ConvBlock(nn.ModelBase):
            def on_build(self, in_planes, out_planes):
                self.in_planes = in_planes
                self.out_planes = out_planes

                self.bn1 = nn.BatchNorm2D(in_planes)
                self.conv1 = nn.Conv2D (in_planes, out_planes//2, kernel_size=3, strides=1, padding='SAME', use_bias=False )

                self.bn2 = nn.BatchNorm2D(out_planes//2)
                self.conv2 = nn.Conv2D (out_planes//2, out_planes//4, kernel_size=3, strides=1, padding='SAME', use_bias=False )

                self.bn3 = nn.BatchNorm2D(out_planes//4)
                self.conv3 = nn.Conv2D (out_planes//4, out_planes//4, kernel_size=3, strides=1, padding='SAME', use_bias=False )

                if self.in_planes != self.out_planes:
                    self.down_bn1 = nn.BatchNorm2D(in_planes)
                    self.down_conv1 = nn.Conv2D (in_planes, out_planes, kernel_size=1, strides=1, padding='VALID', use_bias=False )
                else:
                    self.down_bn1 = None
                    self.down_conv1 = None

            def forward(self, input):
                x = input
                x = self.bn1(x)
                x = tf.nn.relu(x)
                x = out1 = self.conv1(x)

                x = self.bn2(x)
                x = tf.nn.relu(x)
                x = out2 = self.conv2(x)

                x = self.bn3(x)
                x = tf.nn.relu(x)
                x = out3 = self.conv3(x)

                x = tf.concat ([out1, out2, out3], axis=-1)

                if self.in_planes != self.out_planes:
                    downsample = self.down_bn1(input)
                    downsample = tf.nn.relu (downsample)
                    downsample = self.down_conv1 (downsample)
                    x = x + downsample
                else:
                    x = x + input

                return x

        class HourGlass (nn.ModelBase):
            def on_build(self, in_planes, depth):
                self.b1 = ConvBlock (in_planes, 256)
                self.b2 = ConvBlock (in_planes, 256)

                if depth > 1:
                    self.b2_plus = HourGlass(256, depth-1)
                else:
                    self.b2_plus = ConvBlock(256, 256)

                self.b3 = ConvBlock(256, 256)

            def forward(self, input):
                up1 = self.b1(input)

                low1 = tf.nn.avg_pool(input, [1,2,2,1], [1,2,2,1], 'VALID')
                low1 = self.b2 (low1)

                low2 = self.b2_plus(low1)
                low3 = self.b3(low2)

                up2 = nn.upsample2d(low3)

                return up1+up2

        class FAN (nn.ModelBase):
            def __init__(self):
                super().__init__(name='FAN')

            def on_build(self):
                self.conv1 = nn.Conv2D (3, 64, kernel_size=7, strides=2, padding='SAME')
                self.bn1 = nn.BatchNorm2D(64)

                self.conv2 = ConvBlock(64, 128)
                self.conv3 = ConvBlock(128, 128)
                self.conv4 = ConvBlock(128, 256)

                self.m = []
                self.top_m = []
                self.conv_last = []
                self.bn_end = []
                self.l = []
                self.bl = []
                self.al = []
                for i in range(4):
                    self.m += [ HourGlass(256, 4) ]
                    self.top_m += [ ConvBlock(256, 256) ]

                    self.conv_last += [ nn.Conv2D (256, 256, kernel_size=1, strides=1, padding='VALID') ]
                    self.bn_end += [ nn.BatchNorm2D(256) ]

                    self.l += [ nn.Conv2D (256, 68, kernel_size=1, strides=1, padding='VALID') ]

                    if i < 4-1:
                        self.bl += [ nn.Conv2D (256, 256, kernel_size=1, strides=1, padding='VALID') ]
                        self.al += [ nn.Conv2D (68, 256, kernel_size=1, strides=1, padding='VALID') ]

            def forward(self, inp) :
                x, = inp
                x = self.conv1(x)
                x = self.bn1(x)
                x = tf.nn.relu(x)

                x = self.conv2(x)
                x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
                x = self.conv3(x)
                x = self.conv4(x)

                outputs = []
                previous = x
                for i in range(4):
                    ll = self.m[i] (previous)
                    ll = self.top_m[i] (ll)
                    ll = self.conv_last[i] (ll)
                    ll = self.bn_end[i] (ll)
                    ll = tf.nn.relu(ll)
                    tmp_out = self.l[i](ll)
                    outputs.append(tmp_out)
                    if i < 4 - 1:
                        ll = self.bl[i](ll)
                        previous = previous + ll + self.al[i](tmp_out)
                x = outputs[-1]
                x = tf.transpose(x, (0,3,1,2) )
                return x

        e = None
        if place_model_on_cpu:
            e = tf.device("/CPU:0")

        if e is not None: e.__enter__()
        self.model = FAN()
        self.model.load_weights(str(model_path))
        if e is not None: e.__exit__(None,None,None)

        self.model.build_for_run ([ ( tf.float32, (None,256,256,3) ) ])

    def extract (self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        landmarks = []
        for (left, top, right, bottom) in rects:
            scale = (right - left + bottom - top) / 195.0

            center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
            centers = [ center ]

            if multi_sample:
                centers += [ center + [-1,-1],
                             center + [1,-1],
                             center + [1,1],
                             center + [-1,1],
                           ]

            images = []
            ptss = []

            try:
                for c in centers:
                    images += [ self.crop(input_image, c, scale)  ]

                images = np.stack (images)
                images = images.astype(np.float32) / 255.0

                predicted = []
                for i in range( len(images) ):
                    predicted += [ self.model.run ( [ images[i][None,...] ]  )[0] ]

                predicted = np.stack(predicted)

                for i, pred in enumerate(predicted):
                    ptss += [ self.get_pts_from_predict ( pred, centers[i], scale) ]
                pts_img = np.mean ( np.array(ptss), 0 )

                landmarks.append (pts_img)
            except:
                landmarks.append (None)

        if second_pass_extractor is not None:
            for i, lmrks in enumerate(landmarks):
                try:
                    if lmrks is not None:
                        image_to_face_mat = LandmarksProcessor.get_transform_mat (lmrks, 256, FaceType.FULL)
                        face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC )

                        rects2 = second_pass_extractor.extract(face_image, is_bgr=is_bgr)
                        if len(rects2) == 1: #dont do second pass if faces != 1 detected in cropped image
                            lmrks2 = self.extract (face_image, [ rects2[0] ], is_bgr=is_bgr, multi_sample=True)[0]
                            landmarks[i] = LandmarksProcessor.transform_points (lmrks2, image_to_face_mat, True)
                except:
                    pass

        return landmarks

    def transform(self, point, center, scale, resolution):
        pt = np.array ( [point[0], point[1], 1.0] )
        h = 200.0 * scale
        m = np.eye(3)
        m[0,0] = resolution / h
        m[1,1] = resolution / h
        m[0,2] = resolution * ( -center[0] / h + 0.5 )
        m[1,2] = resolution * ( -center[1] / h + 0.5 )
        m = np.linalg.inv(m)
        return np.matmul (m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype( np.int )
        br = self.transform([resolution, resolution], center, scale, resolution).astype( np.int )

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        a_ch, a_h, a_w = a.shape

        b = a.reshape ( (a_ch, a_h*a_w) )
        c = b.argmax(1).reshape ( (a_ch, 1) ).repeat(2, axis=1).astype(np.float)
        c[:,0] %= a_w
        c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a_w), 0, c[:,1] )

        for i in range(a_ch):
            pX, pY = int(c[i,0]), int(c[i,1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
                c[i] += np.sign(diff)*0.25

        c += 0.5

        return np.array( [ self.transform (c[i], center, scale, a_w) for i in range(a_ch) ] )
