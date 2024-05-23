import operator
from pathlib import Path

import cv2
import numpy as np

from core.leras import nn

class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        nn.initialize(data_format="NHWC")
        tf = nn.tf

        model_path = Path(__file__).parent / "S3FD.npy"
        if not model_path.exists():
            raise Exception("Unable to load S3FD.npy")

        class L2Norm(nn.LayerBase):
            def __init__(self, n_channels, **kwargs):
                self.n_channels = n_channels
                super().__init__(**kwargs)

            def build_weights(self):
                self.weight = tf.get_variable ("weight", (1, 1, 1, self.n_channels), dtype=nn.floatx, initializer=tf.initializers.ones )

            def get_weights(self):
                return [self.weight]

            def __call__(self, inputs):
                x = inputs
                x = x / (tf.sqrt( tf.reduce_sum( tf.pow(x, 2), axis=-1, keepdims=True ) ) + 1e-10) * self.weight
                return x

        class S3FD(nn.ModelBase):
            def __init__(self):
                super().__init__(name='S3FD')

            def on_build(self):
                self.minus = tf.constant([104,117,123], dtype=nn.floatx )
                self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, strides=1, padding='SAME')
                self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, strides=1, padding='SAME')

                self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, strides=1, padding='SAME')
                self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, strides=1, padding='SAME')

                self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')

                self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.fc6 = nn.Conv2D(512, 1024, kernel_size=3, strides=1, padding=3)
                self.fc7 = nn.Conv2D(1024, 1024, kernel_size=1, strides=1, padding='SAME')

                self.conv6_1 = nn.Conv2D(1024, 256, kernel_size=1, strides=1, padding='SAME')
                self.conv6_2 = nn.Conv2D(256, 512, kernel_size=3, strides=2, padding='SAME')

                self.conv7_1 = nn.Conv2D(512, 128, kernel_size=1, strides=1, padding='SAME')
                self.conv7_2 = nn.Conv2D(128, 256, kernel_size=3, strides=2, padding='SAME')

                self.conv3_3_norm = L2Norm(256)
                self.conv4_3_norm = L2Norm(512)
                self.conv5_3_norm = L2Norm(512)


                self.conv3_3_norm_mbox_conf = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3_norm_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv4_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv5_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.fc7_mbox_conf = nn.Conv2D(1024, 2, kernel_size=3, strides=1, padding='SAME')
                self.fc7_mbox_loc = nn.Conv2D(1024, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv6_2_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv6_2_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv7_2_mbox_conf = nn.Conv2D(256, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv7_2_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

            def forward(self, inp):
                x, = inp
                x = x - self.minus
                x = tf.nn.relu(self.conv1_1(x))
                x = tf.nn.relu(self.conv1_2(x))
                x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], "VALID")

                x = tf.nn.relu(self.conv2_1(x))
                x = tf.nn.relu(self.conv2_2(x))
                x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], "VALID")

                x = tf.nn.relu(self.conv3_1(x))
                x = tf.nn.relu(self.conv3_2(x))
                x = tf.nn.relu(self.conv3_3(x))
                f3_3 = x
                x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], "VALID")

                x = tf.nn.relu(self.conv4_1(x))
                x = tf.nn.relu(self.conv4_2(x))
                x = tf.nn.relu(self.conv4_3(x))
                f4_3 = x
                x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], "VALID")

                x = tf.nn.relu(self.conv5_1(x))
                x = tf.nn.relu(self.conv5_2(x))
                x = tf.nn.relu(self.conv5_3(x))
                f5_3 = x
                x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], "VALID")

                x = tf.nn.relu(self.fc6(x))
                x = tf.nn.relu(self.fc7(x))
                ffc7 = x

                x = tf.nn.relu(self.conv6_1(x))
                x = tf.nn.relu(self.conv6_2(x))
                f6_2 = x

                x = tf.nn.relu(self.conv7_1(x))
                x = tf.nn.relu(self.conv7_2(x))
                f7_2 = x

                f3_3 = self.conv3_3_norm(f3_3)
                f4_3 = self.conv4_3_norm(f4_3)
                f5_3 = self.conv5_3_norm(f5_3)

                cls1 = self.conv3_3_norm_mbox_conf(f3_3)
                reg1 = self.conv3_3_norm_mbox_loc(f3_3)

                cls2 = tf.nn.softmax(self.conv4_3_norm_mbox_conf(f4_3))
                reg2 = self.conv4_3_norm_mbox_loc(f4_3)

                cls3 = tf.nn.softmax(self.conv5_3_norm_mbox_conf(f5_3))
                reg3 = self.conv5_3_norm_mbox_loc(f5_3)

                cls4 = tf.nn.softmax(self.fc7_mbox_conf(ffc7))
                reg4 = self.fc7_mbox_loc(ffc7)

                cls5 = tf.nn.softmax(self.conv6_2_mbox_conf(f6_2))
                reg5 = self.conv6_2_mbox_loc(f6_2)

                cls6 = tf.nn.softmax(self.conv7_2_mbox_conf(f7_2))
                reg6 = self.conv7_2_mbox_loc(f7_2)

                # max-out background label
                bmax = tf.maximum(tf.maximum(cls1[:,:,:,0:1], cls1[:,:,:,1:2]), cls1[:,:,:,2:3])

                cls1 = tf.concat ([bmax, cls1[:,:,:,3:4] ], axis=-1)
                cls1 = tf.nn.softmax(cls1)

                return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

        e = None
        if place_model_on_cpu:
            e = tf.device("/CPU:0")

        if e is not None: e.__enter__()
        self.model = S3FD()
        self.model.load_weights (model_path)
        if e is not None: e.__exit__(None,None,None)

        self.model.build_for_run ([ ( tf.float32, nn.get4Dshape (None,None,3) ) ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, is_bgr=True, is_remove_intersects=False):

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize (input_image, ( int(w/input_scale), int(h/input_scale) ), interpolation=cv2.INTER_LINEAR)

        olist = self.model.run ([ input_image[None,...] ] )

        detected_faces = []
        for ltrb in self.refine (olist):
            l,t,r,b = [ x*input_scale for x in ltrb]
            bt = b-t
            if min(r-l,bt) < 40: #filtering faces < 40pix by any side
                continue
            b += bt*0.1 #enlarging bottom line a bit for 2DFAN-4, because default is not enough covering a chin
            detected_faces.append ( [int(x) for x in (l,t,r,b) ] )

        #sort by largest area first
        detected_faces = [ [(l,t,r,b), (r-l)*(b-t) ]  for (l,t,r,b) in detected_faces ]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True )
        detected_faces = [ x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range( len(detected_faces)-1, 0, -1):
                l1,t1,r1,b1 = detected_faces[i]
                l0,t0,r0,b0 = detected_faces[i-1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx>=0) and (dy>=0):
                    detected_faces.pop(i)

        return detected_faces

    def refine(self, olist):
        bboxlist = []
        for i, ((ocls,), (oreg,)) in enumerate ( zip ( olist[::2], olist[1::2] ) ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls[...,1] > 0.05)):
                score = ocls[hindex, windex, 1]
                loc   = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)) )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
        bboxlist = [ x[:-1].astype(np.int) for x in bboxlist if x[-1] >= 0.5]
        return bboxlist

    def refine_nms(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
