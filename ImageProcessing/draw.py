"""
找人臉加效果框
"""
import cv2, numpy as np
from ImageProcessing.files import MATERIAL, CASCADE_FILE, im_show
import math
# 特效顏色
EFFECT_BOX_COLOR = (20, 20, 20)
EFFECT_TEXT_COLOR = (100, 100, 100)

# Kmeans常數
K = 30
MAX_ITER = 30


# 改變圖片大小與取得二值圖
def resized_image_and_binary(fileName, size):
    image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image, binary


# 圖片合成
def image_merge(fg, bg, mask, color=None):
    # 取得反遮罩
    mask_inv = cv2.bitwise_not(mask)

    # 畫成純色圖片
    if color is not None:
        rows, cols, _ = bg.shape
        fg = cv2.rectangle(bg.copy(), (0, 0), (cols, rows), color, -1)

    bg = cv2.bitwise_and(bg, bg, mask=mask)
    fg = cv2.bitwise_and(fg, fg, mask=mask_inv)
    output = cv2.add(fg, bg)

    return output


def draw(image, mask):
    Omage = image.copy()
    rows, cols, _ = image.shape
    size = (cols, rows)

    # 灰階圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Kmeans
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITER,
                1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    image = res.reshape((image.shape))

    # 背景模糊
    blur = cv2.GaussianBlur(image.copy(), (21, 21), 0)
    output = image_merge(blur, image, mask)

    # 特效框
    cover, cover_mask = resized_image_and_binary(MATERIAL['comic'], size)
    output = image_merge(cover, output, cover_mask, EFFECT_BOX_COLOR)

    # 特效字體
    gogogo, gogogo_mask = resized_image_and_binary(MATERIAL['gogogo'], size)
    gogogo_mask = cv2.bitwise_or(mask, gogogo_mask)
    output = image_merge(gogogo, output, gogogo_mask, EFFECT_TEXT_COLOR)

    # 分級器 - 抓人臉
    cascade = cv2.CascadeClassifier(CASCADE_FILE['frontalface'])
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=7,
                                     minSize=(60, 60))

    # 防止人臉被特效蓋到
    if np.any(faces):
        # 有抓到人臉的狀況
        for (x, y, w, h) in faces:
            # output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0),
            #                        2)
            output[y:y + h, x:x + w] = image[y:y + h, x:x + w]
            
            
            # 全臉
            # output[y:y + h, x:x + w] = cv2.addWeighted(image[y:y + h, x:x + w], 0.5, Omage[y:y + h, x:x + w], 0.5, 0)
            # output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # 鼻子
            ys = y + int(h * 0.4)
            xs = x + int(w / 3)
            hs = int(h * 0.3)
            ws = int(w / 3)
            output[ys:ys + hs, xs:xs + ws] = cv2.addWeighted(image[ys:ys + hs, xs:xs + ws], 0.6, Omage[ys:ys + hs, xs:xs + ws], 0.4, 0)
            # output = cv2.rectangle(output, (xs , ys ), (xs + ws , ys + hs), (255, 0, 0), 1)

            # 嘴巴
            ys = y + int(h * 0.675)
            xs = x + int(w * 0.25)
            hs = int(h * 0.25)
            ws = int(w * 0.5)
            output[ys:ys + hs, xs:xs + ws] = cv2.addWeighted(image[ys:ys + hs, xs:xs + ws], 0.5, Omage[ys:ys + hs, xs:xs + ws], 0.5, 0)
            # output = cv2.rectangle(output, (xs , ys ), (xs + ws , ys + hs), (255, 0, 0), 1)

           


    else:
        # 沒抓到人臉的狀況
        output = image_merge(output, image, mask)

    im_show(output)
