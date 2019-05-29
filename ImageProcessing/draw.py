"""
找人臉加效果框
"""
import cv2, numpy as np
from ImageProcessing.files import MATERIAL, CASCADE_FILE, im_show

# 特效顏色
EFFECT_BOX_COLOR = (20, 20, 20)
EFFECT_TEXT_COLOR = (100, 100, 100)


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
    rows, cols, _ = image.shape
    size = (cols, rows)

    # 灰階圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

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
            output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0),
                                   2)
            output[y:y + h, x:x + w] = image[y:y + h, x:x + w]
    else:
        # 沒抓到人臉的狀況
        output = image_merge(output, image, mask)

    im_show(output)
