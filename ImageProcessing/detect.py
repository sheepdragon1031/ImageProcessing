"""
找人臉加效果框
"""
import cv2
from math import floor
from .picture import MATERIAL, CASCADE_FILE


def detect(filename):
    image = filename[0]
    cascade = cv2.CascadeClassifier(CASCADE_FILE['frontalface'])
    # image = cv2.imread(filename)
    cover = cv2.imread(MATERIAL['comic'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rows, cols, channels = image.shape
    #分級器
    faces = cascade.detectMultiScale(
        gray,
        # detector options
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(60, 60))
    i = 0
    oimage = image.copy()

    #特效框
    cover = cv2.resize(cover, (cols, rows), interpolation=cv2.INTER_CUBIC)
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(cover, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img = image[0:rows, 0:cols]
    bcg = cover[0:rows, 0:cols]

    img1_bg = cv2.bitwise_and(img, img, mask=mask)
    saveImg = img1_bg.copy()

    #防止人臉覆蓋
    for (x, y, w, h) in faces:
        i += 1
        new = cv2.rectangle(saveImg, (0, 0), (cols, rows), (0, 0, 0), -1)
        nx = floor(x + w * 0.5)
        ny = floor(y + h * 0.5)
        w = floor(w * 0.5)
        h = floor(h * 0.6)
        new = cv2.ellipse(new, (nx, ny), (w, h), 0, 0, 360, (255, 255, 255),
                          -1)

        new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(new, 50, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frows, fcols, fchannels = oimage.shape
        img1 = oimage[0:frows, 0:fcols]
        img2 = img1_bg[0:frows, 0:fcols]

        img1_bg = cv2.bitwise_and(img1, img1, mask=mask)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

        saveImg = cv2.add(img1_bg, img2_fg)
        # 用不到功能
        # cv2.rectangle(saveImg, (x , y), (x + w , y + h), (255, 255, 255), 2)
        # cv2.ellipse(saveImg, (nx, ny), ( w, h), 0, 0, 360, (255, 255, 255), 2)
        # temp=image[y:y+h,x:x+w,:]
        # cv2.imwrite('%s_%d.jpg'%(path.basename(filename).split('.')[0],i),temp)

    #防止方法人臉覆蓋

    if i == 0:
        img1 = oimage[0:rows, 0:cols]
        img2 = saveImg[0:rows, 0:cols]
        mask = filename[1]
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(img1, img1, mask=mask)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
        saveImg = cv2.add(img1_bg, img2_fg)

    cv2.imshow("image", saveImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #寫入
    #cv2.imwrite("out.png", image)
