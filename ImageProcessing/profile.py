"""
找出人體位置
"""
import cv2
from os import path
from ImageProcessing.fillHole import fillHole
from ImageProcessing.files import MATERIAL, imShow


def profile(filename):
    if not path.isfile(filename):
        raise RuntimeError("%s: not found" % filename)

    image = cv2.imread(filename)
    cover = cv2.imread(MATERIAL['gogogo'])

    #轉換灰度並去噪聲
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray) //顏色均質化
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    blurred = cv2.GaussianBlur(blurred, (9, 9), 0)

    # 提取圖像的梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)

    # gradX = cv2.Sobel( blurred , ddepth = cv2 . CV_32F , dx = 1 , dy = 0 )
    # gradY = cv2.Sobel( blurred , ddepth = cv2 . CV_32F , dx = 0 , dy = 1 )
    # gradients = cv2.subtract( gradX , gradY )
    # gradients = cv2.convertScaleAbs( gradients )

    # blurred = cv2.GaussianBlur ( gradients ,  ( 9 ,  9 ) , 0 )
    # cv2.imshow('closeds',blurred)

    # cv2.imshow( "gradients",blurred)
    (_, thresh) = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (9, 9), 0)

    # edges = cv2.Canny(blurred,200,200)
    closed = cv2.dilate(thresh, None, iterations=36)
    closed = cv2.erode(closed, None, iterations=36)
    closed = cv2.dilate(closed, None, iterations=10)
    # gray = cv2.equalizeHist(closed)

    for i in range(0, closed.shape[0]):
        for j in range(0, closed.shape[1]):
            pixel = closed.item(i, j)
            if (pixel < 70):
                closed.itemset((i, j), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    # mix = cv2.addWeighted(thresh,0.5,closed,0.5,0)
    MixMask = fillHole(closed)

    images = image.copy()
    rows, cols, channels = images.shape
    images = cv2.GaussianBlur(images, (9, 9), 0)
    images = cv2.GaussianBlur(images, (9, 9), 0)
    images = cv2.blur(images, (9, 9))
    img = image[0:rows, 0:cols]
    bcg = images[0:rows, 0:cols]
    mask = MixMask
    mask_inv = cv2.bitwise_not(MixMask)
    # mask = cv2.erode ( mask , None , iterations = 3 )

    img1_bg = cv2.bitwise_and(img, img, mask=mask)
    img2_fg = cv2.bitwise_and(bcg, bcg, mask=mask_inv)
    dst = cv2.addWeighted(img1_bg, 1, img2_fg, 1, 0)
    # img2_fg = cv2.dilate ( img2_fg , None , iterations = 3 )

    return [dst, mask]
