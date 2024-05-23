import math

import cv2
import numpy as np
import numpy.linalg as npla

from .umeyama import umeyama


def get_power_of_two(x):
    i = 0
    while (1 << i) < x:
        i += 1
    return i

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

def polygon_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def rotate_point(origin, point, deg):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    rad = deg * math.pi / 180.0
    qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
    qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
    return np.float32([qx, qy])
    
def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform (mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

    
def transform_mat(mat, res, tx, ty, rotation, scale):
    """
    transform mat in local space of res
    scale -> translate -> rotate
    
        tx,ty       float
        rotation    int degrees
        scale       float
    """
    
    
    lt, rt, lb, ct = transform_points (  np.float32([(0,0),(res,0),(0,res),(res / 2, res/2) ]),mat, True)
    
    hor_v = (rt-lt).astype(np.float32)
    hor_size = npla.norm(hor_v)
    hor_v /= hor_size
    
    ver_v = (lb-lt).astype(np.float32)
    ver_size = npla.norm(ver_v)
    ver_v /= ver_size
    
    bt_diag_vec = (rt-ct).astype(np.float32)
    half_diag_len = npla.norm(bt_diag_vec)
    bt_diag_vec /= half_diag_len
    
    tb_diag_vec = np.float32( [ -bt_diag_vec[1], bt_diag_vec[0] ]  )

    rt = ct + bt_diag_vec*half_diag_len*scale 
    lb = ct - bt_diag_vec*half_diag_len*scale
    lt = ct - tb_diag_vec*half_diag_len*scale
    
    rt[0] += tx*hor_size
    lb[0] += tx*hor_size
    lt[0] += tx*hor_size
    rt[1] += ty*ver_size
    lb[1] += ty*ver_size
    lt[1] += ty*ver_size
    
    rt = rotate_point(ct, rt, rotation)
    lb = rotate_point(ct, lb, rotation)
    lt = rotate_point(ct, lt, rotation)
    
    return cv2.getAffineTransform( np.float32([lt, rt, lb]), np.float32([ [0,0], [res,0], [0,res] ]) )
