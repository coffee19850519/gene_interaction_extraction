import numpy as np
import cv2,math


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    cv2.affine
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
      center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def getPointAffinedPos(src_points, center, angle):

    dx = src_points[0] - center[0]
    dy = src_points[1] - center[1]

    dst_x = cv2.cvRound(dx * math.cos(angle) + dy * math.sin(angle) + center[0])
    dst_y = cv2.cvRound(-dx * math.sin(angle) + dy * math.cos(angle) + center[1])
    return (dst_x, dst_y)


if __name__== "__main__":

    img_file = r'C:\Users\LSC-110\Desktop\test\pri-mir-150_EP300.png'









