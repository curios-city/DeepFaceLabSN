import cv2

def label_face_filename(face, filename):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (5, face.shape[0] - 10)
    thickness = 1
    fontScale = 0.5
    color = (255, 255, 255)
    face = face.copy() # numpy array issue
    cv2.putText(face, filename, org, font, fontScale, color, thickness, cv2.LINE_AA)

    return face
