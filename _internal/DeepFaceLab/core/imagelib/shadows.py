from typing import Counter
import numpy as np
import cv2

# https://github.com/OsamaMazhar/Random-Shadows-Highlights
# img is in format np 0-1, float
def shadow_highlights_augmentation(img, high_ratio=(1, 2.5), low_ratio=(0.2, 0.6), seed=None):
    rnd_state = np.random.RandomState (seed)

    left_low_ratio = (0.2, 0.6)
    left_high_ratio = (0, 0.2)
    right_low_ratio = (0.3, 0.6)
    right_high_ratio = (0, 0.2)

    # check
    img = np.clip(img*255, 0, 255).astype(np.uint8)

    w, h, _ = img.shape

    high_bright_factor = rnd_state.uniform(high_ratio[0], high_ratio[1])
    low_bright_factor = rnd_state.uniform(low_ratio[0], low_ratio[1])

    left_low_factor = rnd_state.uniform(left_low_ratio[0]*h, left_low_ratio[1]*h)
    left_high_factor = rnd_state.uniform(left_high_ratio[0]*h, left_high_ratio[1]*h)
    right_low_factor = rnd_state.uniform(right_low_ratio[0]*h, right_low_ratio[1]*h)
    right_high_factor = rnd_state.uniform(right_high_ratio[0]*h, right_high_ratio[1]*h)

    tl = (-20, left_high_factor)
    bl = (-20, left_high_factor+left_low_factor)

    tr = (w, right_high_factor)
    br = (w, right_high_factor+right_low_factor)

    contour = np.array([tl, tr, br, bl], dtype=np.int32)

    rnd_angle = rnd_state.uniform(0, 359)
    contour = rotate_contour(contour, rnd_angle)

    mask = np.zeros(img.shape, dtype=img.dtype)
    cv2.fillPoly(mask, [contour], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    inverted_mask = cv2.bitwise_not(mask)

    # blur inverted mask with random intensity
    inverted_mask = cv2.GaussianBlur(inverted_mask, (0,0), sigmaX=rnd_state.randint(3, 10), borderType = cv2.BORDER_DEFAULT)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=rnd_state.randint(3, 10), borderType = cv2.BORDER_DEFAULT)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.multiply(hsv[..., 2], high_bright_factor)
    high_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv[..., 2] = cv2.multiply(hsv[..., 2], low_bright_factor)
    low_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    for i in range(3):
        img[:, :, i] = img[:, :, i] * (mask/255) + high_brightness[:, :, i] * (1-mask/255)
        img[:, :, i] = img[:, :, i] * (inverted_mask/255) + low_brightness[:, :, i] * (1-inverted_mask/255)

    img = np.clip(img/255.0, 0, 1).astype(np.float32)

    return img


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    # cnt = cv2.fromarray(cnt.copy())
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    coordinates = cnt_norm[:, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0] = xs
    cnt_norm[:, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated