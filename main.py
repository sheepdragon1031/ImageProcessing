import numpy as np
import numpy
import cv2
import os.path
import math

# 填洞方法
def fillHole(im_in):
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv

    return im_out

# 找人臉加效果框
def detect(filename, cascade_file = "haarcascade_frontalface_default.xml"):
    image = filename[0]
    cascade = cv2.CascadeClassifier(cascade_file)
    # image = cv2.imread(filename)
    cover = cv2.imread('./material/comic.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rows,cols,channels = image.shape
    #分級器
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 7,
                                     minSize = (50, 50))
    i=0
    oimage = image.copy()
    
    #特效框
    cover =  cv2.resize(cover, (cols ,rows), interpolation=cv2.INTER_CUBIC)
    cover = cv2.cvtColor(cover,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(cover, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
   
    img = image[0:rows, 0:cols ]
    bcg = cover[0:rows, 0:cols ]

    
    img1_bg = cv2.bitwise_and(img,img,mask = mask)
    saveImg = img1_bg.copy()

    #防止人臉覆蓋
    for (x, y, w, h) in faces:
        i+=1
        new = cv2.rectangle(oimage.copy(), (0 , 0), (cols , rows), (0, 0, 0), -1)
        nx = math.floor(x + w * 0.5)
        ny = math.floor(y + h * 0.5)
        w = math.floor(w * 0.5)
        h = math.floor(h * 0.7)
        new = cv2.ellipse(new, (nx, ny), ( w, h), 0, 0, 360, (255, 255, 255), -1)
        
        new = cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(new, 50, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frows,fcols,fchannels = oimage.shape
        img1 = oimage[0:frows, 0:fcols ]
        img2 = img1_bg[0:frows, 0:fcols ]
        
        img1_bg = cv2.bitwise_and(img1,img1,mask = mask)
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
        saveImg = cv2.add(img1_bg,img2_fg)
        # 用不到功能
        # cv2.rectangle(image, (x , y), (x + w , y + h), (255, 255, 255), 2)
        # temp=image[y:y+h,x:x+w,:]
        # cv2.imwrite('%s_%d.jpg'%(os.path.basename(filename).split('.')[0],i),temp)
    
    #防止方法人臉覆蓋
    if i == 0:
        img1 = oimage[0:rows, 0:cols ]
        img2 = saveImg[0:rows, 0:cols ]
        mask = filename[1]
        mask_inv =  cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(img1,img1,mask = mask)
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
        saveImg = cv2.add(img1_bg,img2_fg)
        
    cv2.imshow("image", saveImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #寫入
    #cv2.imwrite("out.png", image)


def profile(filename):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    image = cv2.imread(filename)
    cover = cv2.imread('./material/gogo.png')

  
    #轉換灰度並去噪聲
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray) //顏色均質化
    blurred = cv2.GaussianBlur( gray ,  ( 9 ,  9 ) , 0 )
    blurred = cv2.GaussianBlur( blurred ,  ( 9 ,  9 ) , 0 )
  
    # 提取圖像的梯度
    gradX = cv2.Sobel( blurred , ddepth = cv2 . CV_32F , dx = 1 , dy = 0 ) 
    gradY = cv2.Sobel( blurred , ddepth = cv2 . CV_32F , dx = 0 , dy = 1 )
    gradient = cv2.subtract( gradX , gradY )
    gradient = cv2.convertScaleAbs( gradient )

    blurred = cv2.GaussianBlur ( gradient ,  ( 9 ,  9 ) , 0 ) 
    gradX = cv2.Sobel( blurred , ddepth = cv2 . CV_32F , dx = 1 , dy = 0 ) 
    gradY = cv2.Sobel( blurred , ddepth = cv2 . CV_32F , dx = 0 , dy = 1 )
    gradients = cv2.subtract( gradX , gradY ) 
    gradients = cv2.convertScaleAbs( gradients )


    blurred = cv2.GaussianBlur ( gradients ,  ( 9 ,  9 ) , 0 ) 
    # cv2.imshow( "gradients",blurred)
    ( _ , thresh )  = cv2.threshold ( blurred ,  10 ,  255 , cv2.THRESH_BINARY )
    thresh = cv2.GaussianBlur ( thresh ,  ( 9 ,  9 ) , 0 ) 
    

    # edges = cv2.Canny(blurred,200,200)
    closed = cv2.dilate ( thresh , None , iterations = 36 )
    closed = cv2.erode ( closed , None , iterations = 36 ) 
    closed = cv2.dilate ( closed , None , iterations = 10 )
    # gray = cv2.equalizeHist(closed)
    
    for i in range(0,closed.shape[0]):
        for j in range(0,closed.shape[1]):
            pixel = closed.item(i, j)
            if(pixel < 100):
                closed.itemset((i, j),0)
    

    kernel = cv2.getStructuringElement ( cv2 . MORPH_ELLIPSE ,  ( 40 ,  40 ) ) 
    thresh = cv2.morphologyEx ( thresh , cv2 . MORPH_CLOSE , kernel )
    closed = cv2.morphologyEx ( closed , cv2 . MORPH_CLOSE , kernel )
   
    mix = cv2.addWeighted(thresh,0.5,closed,0.5,0)
    MixMask = fillHole(mix)
    
    images = image.copy()
    rows,cols,channels = images.shape
    images = cv2.GaussianBlur ( images ,  ( 9 ,  9 ) , 0 )
    images = cv2.GaussianBlur ( images ,  ( 9 ,  9 ) , 0 )
    images = cv2.blur(images, (9, 9))
    img = image[0:rows, 0:cols ]
    bcg = images[0:rows, 0:cols ]
    mask = MixMask
    mask_inv = cv2.bitwise_not(MixMask)
    # mask = cv2.erode ( mask , None , iterations = 3 ) 
    
    img1_bg = cv2.bitwise_and(img,img,mask = mask)
    img2_fg = cv2.bitwise_and(bcg,bcg,mask = mask_inv)
    # img2_fg = cv2.dilate ( img2_fg , None , iterations = 3 ) 
    
    # 特效文字
    
    cover = cv2.cvtColor(cover,cv2.COLOR_BGR2GRAY)
    cover =  cv2.resize(cover, (cols ,rows), interpolation=cv2.INTER_CUBIC)
    rets, masks = cv2.threshold(cover, 50, 255, cv2.THRESH_BINARY)
    mask_re = cv2.bitwise_not(masks)

    

    img = img2_fg[0:rows, 0:cols ]
    

    # 人像背景抽離
    img2 = cv2.cvtColor(img2_fg.copy(),cv2.COLOR_BGR2GRAY)
    reti, maski = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    mask_i = cv2.bitwise_not(maski)

    maski = cv2.bitwise_and(masks,maski)
    mask_i = cv2.bitwise_not(maski)


    img1 = cv2.bitwise_and(img,img,mask = maski)
    # 正確的

    cgt = cv2.rectangle(image.copy(), (0 , 0), (cols , rows), (40, 40, 40), -1)
    tcg = cgt[0:rows, 0:cols ]
    remask = cv2.bitwise_xor(maski,mask_inv)
    remasks = cv2.bitwise_not(remask)
    img2 = cv2.bitwise_and(tcg,tcg,mask = remask)
   
    dst = cv2.addWeighted(img1, 1 ,img1_bg, 1, 0)
    dst = cv2.addWeighted(dst, 1 ,img2, 1, 0)
   
    
    return [dst,mask] 
   

def main():
    detect(profile('2.jpg'))
    
 
if __name__ == '__main__':
    main()


