import numpy as np
from .blursharpen import LinearMotionBlur, blursharpen
import cv2

def apply_random_rgb_levels(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
    np_rnd = rnd_state.rand

    inBlack  = np.array([np_rnd()*0.25    , np_rnd()*0.25    , np_rnd()*0.25], dtype=np.float32)
    inWhite  = np.array([1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25], dtype=np.float32)
    inGamma  = np.array([0.5+np_rnd(), 0.5+np_rnd(), 0.5+np_rnd()], dtype=np.float32)

    outBlack  = np.array([np_rnd()*0.25    , np_rnd()*0.25    , np_rnd()*0.25], dtype=np.float32)
    outWhite  = np.array([1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25], dtype=np.float32)

    result = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 1 )
    result = ( result ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
    result = np.clip(result, 0, 1)

    if mask is not None:
        result = img*(1-mask) + result*mask

    return result

def apply_random_hsv_shift(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random

    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    h = ( h + rnd_state.randint(360) ) % 360
    s = np.clip ( s + rnd_state.random()-0.5, 0, 1 )
    v = np.clip ( v + rnd_state.random()-0.5, 0, 1 )

    result = np.clip( cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR) , 0, 1 )
    if mask is not None:
        result = img*(1-mask) + result*mask

    return result

def apply_random_sharpen( img, chance, kernel_max_size, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    sharp_rnd_kernel = rnd_state.randint(kernel_max_size)+1

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        if rnd_state.randint(2) == 0:
            result = blursharpen(result, 1, sharp_rnd_kernel, rnd_state.randint(10) )
        else:
            result = blursharpen(result, 2, sharp_rnd_kernel, rnd_state.randint(50) )
                
        if mask is not None:
            result = img*(1-mask) + result*mask

    return result

def apply_random_motion_blur( img, chance, mb_max_size, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    mblur_rnd_kernel = rnd_state.randint(mb_max_size)+1
    mblur_rnd_deg    = rnd_state.randint(360)

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        result = LinearMotionBlur (result, mblur_rnd_kernel, mblur_rnd_deg )
        if mask is not None:
            result = img*(1-mask) + result*mask

    return result

def apply_random_gaussian_blur( img, chance, kernel_max_size, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        gblur_rnd_kernel = rnd_state.randint(kernel_max_size)*2+1
        result = cv2.GaussianBlur(result, (gblur_rnd_kernel,)*2 , 0)
        if mask is not None:
            result = img*(1-mask) + result*mask

    return result

def apply_random_resize( img, chance, max_size_per, interpolation=cv2.INTER_LINEAR, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        h,w,c = result.shape

        trg = rnd_state.rand()
        rw = w - int( trg * int(w*(max_size_per/100.0)) )
        rh = h - int( trg * int(h*(max_size_per/100.0)) )

        result = cv2.resize (result, (rw,rh), interpolation=interpolation )
        result = cv2.resize (result, (w,h), interpolation=interpolation )
        if mask is not None:
            result = img*(1-mask) + result*mask

    return result

def apply_random_nearest_resize( img, chance, max_size_per, mask=None, rnd_state=None ):
    return apply_random_resize( img, chance, max_size_per, interpolation=cv2.INTER_NEAREST, mask=mask, rnd_state=rnd_state )

def apply_random_bilinear_resize( img, chance, max_size_per, mask=None, rnd_state=None ):
    return apply_random_resize( img, chance, max_size_per, interpolation=cv2.INTER_LINEAR, mask=mask, rnd_state=rnd_state )

def apply_random_jpeg_compress( img, chance, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        h,w,c = result.shape

        quality = rnd_state.randint(10,101)

        ret, result = cv2.imencode('.jpg', np.clip(img*255, 0,255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality] )
        if ret == True:
            result = cv2.imdecode(result, flags=cv2.IMREAD_UNCHANGED)
            result = result.astype(np.float32) / 255.0
            if mask is not None:
                result = img*(1-mask) + result*mask

    return result
    
def apply_random_overlay_triangle( img, max_alpha, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    h,w,c = img.shape
    pt1 = [rnd_state.randint(w), rnd_state.randint(h) ]
    pt2 = [rnd_state.randint(w), rnd_state.randint(h) ]
    pt3 = [rnd_state.randint(w), rnd_state.randint(h) ]
    
    alpha = rnd_state.uniform()*max_alpha
    
    tri_mask = cv2.fillPoly( np.zeros_like(img), [ np.array([pt1,pt2,pt3], np.int32) ], (alpha,)*c )
    
    if rnd_state.randint(2) == 0:
        result = np.clip(img+tri_mask, 0, 1)
    else:
        result = np.clip(img-tri_mask, 0, 1)
    
    if mask is not None:
        result = img*(1-mask) + result*mask

    return result
    
def _min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = min(s1, s0)
    raw_max = min(x.shape[0], x.shape[1])
    return cv2.resize(x, (s1, s0), interpolation=cv2.INTER_LANCZOS4)
    
def _d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y
    
def _get_image_gradient(dist):
    cols = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]))
    rows = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]))
    return cols, rows

def _generate_lighting_effects(content):
    h512 = content
    h256 = cv2.pyrDown(h512)
    h128 = cv2.pyrDown(h256)
    h64 = cv2.pyrDown(h128)
    h32 = cv2.pyrDown(h64)
    h16 = cv2.pyrDown(h32)
    c512, r512 = _get_image_gradient(h512)
    c256, r256 = _get_image_gradient(h256)
    c128, r128 = _get_image_gradient(h128)
    c64, r64 = _get_image_gradient(h64)
    c32, r32 = _get_image_gradient(h32)
    c16, r16 = _get_image_gradient(h16)
    c = c16
    c = _d_resize(cv2.pyrUp(c), c32.shape) * 4.0 + c32
    c = _d_resize(cv2.pyrUp(c), c64.shape) * 4.0 + c64
    c = _d_resize(cv2.pyrUp(c), c128.shape) * 4.0 + c128
    c = _d_resize(cv2.pyrUp(c), c256.shape) * 4.0 + c256
    c = _d_resize(cv2.pyrUp(c), c512.shape) * 4.0 + c512
    r = r16
    r = _d_resize(cv2.pyrUp(r), r32.shape) * 4.0 + r32
    r = _d_resize(cv2.pyrUp(r), r64.shape) * 4.0 + r64
    r = _d_resize(cv2.pyrUp(r), r128.shape) * 4.0 + r128
    r = _d_resize(cv2.pyrUp(r), r256.shape) * 4.0 + r256
    r = _d_resize(cv2.pyrUp(r), r512.shape) * 4.0 + r512
    coarse_effect_cols = c
    coarse_effect_rows = r
    EPS = 1e-10

    max_effect = np.max((coarse_effect_cols**2 + coarse_effect_rows**2)**0.5, axis=0, keepdims=True, ).max(1, keepdims=True)
    coarse_effect_cols = (coarse_effect_cols + EPS) / (max_effect + EPS)
    coarse_effect_rows = (coarse_effect_rows + EPS) / (max_effect + EPS)

    return np.stack([ np.zeros_like(coarse_effect_rows), coarse_effect_rows, coarse_effect_cols], axis=-1)
    
def apply_random_relight(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
        
    def_img = img
        
    if rnd_state.randint(2) == 0:
        light_pos_y = 1.0 if rnd_state.randint(2) == 0 else -1.0
        light_pos_x = rnd_state.uniform()*2-1.0
    else:
        light_pos_y = rnd_state.uniform()*2-1.0
        light_pos_x = 1.0 if rnd_state.randint(2) == 0 else -1.0
                    
    light_source_height = 0.3*rnd_state.uniform()*0.7
    light_intensity = 1.0+rnd_state.uniform()
    ambient_intensity = 0.5
    
    light_source_location = np.array([[[light_source_height, light_pos_y, light_pos_x ]]], dtype=np.float32)
    light_source_direction = light_source_location / np.sqrt(np.sum(np.square(light_source_location)))

    lighting_effect = _generate_lighting_effects(img)
    lighting_effect = np.sum(lighting_effect * light_source_direction, axis=-1).clip(0, 1)
    lighting_effect = np.mean(lighting_effect, axis=-1, keepdims=True)

    result = def_img * (ambient_intensity + lighting_effect * light_intensity) #light_source_color
    result = np.clip(result, 0, 1)
    
    if mask is not None:
        result = def_img*(1-mask) + result*mask
    
    return result