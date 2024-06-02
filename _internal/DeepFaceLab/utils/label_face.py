import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def label_face_filename(face, filename):
    org = (5, face.shape[0] - 15)
    thickness = 1
    fontScale = 0.5
    color = (0, 128, 255)
    face = face.copy() # numpy array issue
    #cv2.putText(face, filename, org, font, fontScale, color, thickness, cv2.LINE_AA)
    face_uint8 = (face*255).astype('uint8')
    imgPIL = Image.fromarray(cv2.cvtColor(face_uint8, cv2.COLOR_BGR2RGB))
    drawPIL = ImageDraw.Draw(imgPIL)
    textSize = 15
    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    text = filename
    drawPIL.text(org, text, color, font=fontText)
    imgPutText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)
    img_float32 = imgPutText.astype(np.float32)/ 255.0
    return img_float32
