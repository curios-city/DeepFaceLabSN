import numpy as np
import cv2
from scipy.spatial import Delaunay


def applyAffineTransform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    return cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

def morphTriangle(dst_img, src_img, st, dt) :
    (h,w,c) = dst_img.shape
    sr = np.array( cv2.boundingRect(np.float32(st)) )
    dr = np.array( cv2.boundingRect(np.float32(dt)) )
    sRect = st - sr[0:2]
    dRect = dt - dr[0:2]
    d_mask = np.zeros((dr[3], dr[2], c), dtype = np.float32)
    cv2.fillConvexPoly(d_mask, np.int32(dRect), (1.0,)*c, 8, 0);
    imgRect = src_img[sr[1]:sr[1] + sr[3], sr[0]:sr[0] + sr[2]]
    size = (dr[2], dr[3])
    warpImage1 = applyAffineTransform(imgRect, sRect, dRect, size)

    if c == 1:
        warpImage1 = np.expand_dims( warpImage1, -1 )

    dst_img[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]] = dst_img[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]]*(1-d_mask) + warpImage1 * d_mask

def morph_by_points (image, sp, dp):
    if sp.shape != dp.shape:
        raise ValueError ('morph_by_points() sp.shape != dp.shape')
    (h,w,c) = image.shape

    result_image = np.zeros(image.shape, dtype = image.dtype)

    for tri in Delaunay(dp).simplices:
        morphTriangle(result_image, image, sp[tri], dp[tri])

    return result_image