import numpy as np
import cv2

def draw_polygon (image, points, color, thickness = 1):
    points_len = len(points)
    for i in range (0, points_len):
        p0 = tuple( points[i] )
        p1 = tuple( points[ (i+1) % points_len] )
        cv2.line (image, p0, p1, color, thickness=thickness)

def draw_rect(image, rect, color, thickness=1):
    l,t,r,b = rect
    draw_polygon (image, [ (l,t), (r,t), (r,b), (l,b ) ], color, thickness)
