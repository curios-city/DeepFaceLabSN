import numpy as np

def random_crop(img, w, h):
    height, width = img.shape[:2]
    
    h_rnd = height - h
    w_rnd = width - w
    
    y = np.random.randint(0, h_rnd) if h_rnd > 0 else 0
    x = np.random.randint(0, w_rnd) if w_rnd > 0 else 0
    
    return img[y:y+height, x:x+width]
                        
def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")

    if c == 0 and target_channels > 0:
        img = img[...,np.newaxis]
        c = 1

    if c == 1 and target_channels > 1:
        img = np.repeat (img, target_channels, -1)
        c = target_channels

    if c > target_channels:
        img = img[...,0:target_channels]
        c = target_channels

    return img

def cut_odd_image(img):
    h, w, c = img.shape
    wm, hm = w % 2, h % 2
    if wm + hm != 0:
        img = img[0:h-hm,0:w-wm,:]
    return img

def overlay_alpha_image(img_target, img_source, xy_offset=(0,0) ):
    (h,w,c) = img_source.shape
    if c != 4:
        raise ValueError("overlay_alpha_image, img_source must have 4 channels")

    x1, x2 = xy_offset[0], xy_offset[0] + w
    y1, y2 = xy_offset[1], xy_offset[1] + h

    alpha_s = img_source[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img_target[y1:y2, x1:x2, c] = (alpha_s * img_source[:, :, c] +
                                        alpha_l * img_target[y1:y2, x1:x2, c])