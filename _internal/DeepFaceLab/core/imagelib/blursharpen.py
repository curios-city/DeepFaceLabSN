import cv2
import numpy as np

def LinearMotionBlur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
    k = k * ( 1.0 / np.sum(k) )
    return cv2.filter2D(image, -1, k)
    
def blursharpen (img, sharpen_mode=0, kernel_size=3, amount=100):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if amount > 0:
        if sharpen_mode == 1: #box
            kernel = np.zeros( (kernel_size, kernel_size), dtype=np.float32)
            kernel[ kernel_size//2, kernel_size//2] = 1.0
            box_filter = np.ones( (kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
            kernel = kernel + (kernel - box_filter) * amount
            return cv2.filter2D(img, -1, kernel)
        elif sharpen_mode == 2: #gaussian
            blur = cv2.GaussianBlur(img, (kernel_size, kernel_size) , 0)
            img = cv2.addWeighted(img, 1.0 + (0.5 * amount), blur, -(0.5 * amount), 0)
            return img
        elif sharpen_mode == 3: #unsharpen_mask
            img = unsharpen_mask(img, amount=amount)
    elif amount < 0:
        n = -amount
        while n > 0:

            img_blur = cv2.medianBlur(img, 5)
            if int(n / 10) != 0:
                img = img_blur
            else:
                pass_power = (n % 10) / 10.0
                img = img*(1.0-pass_power)+img_blur*pass_power
            n = max(n-10,0)

        return img
    return img
    
def gaussian_sharpen(img, amount=100, sigma=1.0):
    img =  cv2.addWeighted(img, 1.0 + (0.05 * amount), cv2.GaussianBlur(img, (0, 0), sigma), -(0.05 * amount), 0)
    return img
    
def unsharpen_mask(img, amount=100, sigma=0.0, threshold = (5.0 / 255.0)):
    radius = max(1, round(img.shape[0] * (amount / 100)))
    kernel_size = int((radius * 2) + 1)
    kernel_size = (kernel_size, kernel_size)
    blur = cv2.GaussianBlur(img, kernel_size, sigma)
    low_contrast_mask = (abs(img - blur) < threshold).astype("float32")
    sharpened = (img * (1.0 + (0.05 * amount))) + (blur * -(0.05 * amount))
    img = (img * (1.0 - low_contrast_mask)) + (sharpened * low_contrast_mask)
    return img