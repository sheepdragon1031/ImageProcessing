import numpy as np
import numpy
import cv2
import os.path
import math


# img=cv2.imread('1.jpg',cv2.IMREAD_UNCHANGED)
# cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('img',img)
# cv2.waitKey(0)
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


def detect(filename, cascade_file = "haarcascade_frontalface_default.xml"):
    # if not os.path.isfile(cascade_file):
    #     raise RuntimeError("%s: not found" % cascade_file)
    image = filename
    cascade = cv2.CascadeClassifier(cascade_file)
    # image = cv2.imread(filename)
    cover = cv2.imread('./material/comic.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rows,cols,channels = image.shape
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 7,
                                     minSize = (50, 50))
    i=0
    oimage = image.copy()
    
    cover =  cv2.resize(cover, (cols ,rows), interpolation=cv2.INTER_CUBIC)
    cover = cv2.cvtColor(cover,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(cover, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
   
    img = image[0:rows, 0:cols ]
    bcg = cover[0:rows, 0:cols ]

    
    img1_bg = cv2.bitwise_and(img,img,mask = mask)
    saveImg = img1_bg.copy()
    for (x, y, w, h) in faces:
        i+=1
        new = cv2.rectangle(oimage.copy(), (0 , 0), (cols , rows), (0, 0, 0), -1)
        nx = math.floor(x + w * 0.5)
        ny = math.floor(y + h * 0.5)
        w = math.floor(w * 0.5)
        h = math.floor(h * 0.5)
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
        # cv2.rectangle(image, (x , y), (x + w , y + h), (255, 255, 255), 2)
        # temp=image[y:y+h,x:x+w,:]
        # cv2.imwrite('%s_%d.jpg'%(os.path.basename(filename).split('.')[0],i),temp)
    cv2.imshow("image", saveImg)
    
    # dst = cv2.addWeighted(img2_fg, 1 ,img1_bg, 1, 0)
    # cv2.imshow("FaceDetect", dst)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite("out.png", image)


def profile(filename):
    image = cv2.imread(filename)
    itext = cv2.imread('./material/gogo.png')

    #image = cv2.imread('14.png')
    #cv2.imshow( "image",image)

    #轉換灰度並去噪聲
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray) //顏色均質化
    blurred = cv2.GaussianBlur( gray ,  ( 9 ,  9 ) , 0 )
    blurred = cv2.GaussianBlur( blurred ,  ( 9 ,  9 ) , 0 )
    
    # kernels = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # Sharpen = cv2.filter2D(blurred, -1, kernels)
    
    # blurred = cv2.GaussianBlur( blurred ,  ( 9 ,  9 ) , 0 )
    #cv2.imshow( "thresh",blurred)
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
    # cv2.imshow( "fillHoleclosed",MixMask)
    images = image.copy()
    trows,tcols,tchannels = itext.shape
    rows,cols,channels = images.shape
    images = cv2.GaussianBlur ( images ,  ( 9 ,  9 ) , 0 )
    images = cv2.GaussianBlur ( images ,  ( 9 ,  9 ) , 0 )
    images = cv2.blur(images, (9, 9))
    img = image[0:rows, 0:cols ]
    bcg = images[0:rows, 0:cols ]
    tcg =  itext[0: trows, 0:tcols]
    mask = MixMask
    mask_inv = cv2.bitwise_not(MixMask)
    mask = cv2.erode ( mask , None , iterations = 3 ) 
    # mask = cv2.GaussianBlur ( MixMask ,  ( 9 ,  9 ) , 0 )
    
    # mask = cv2.GaussianBlur ( mask ,  ( 9 ,  9 ) , 0 )
    # mask = cv2.erode ( mask , None , iterations = 2 )
    new = cv2.rectangle(image.copy(), (0 , 0), (cols , rows), (0, 0, 0), -1)
    nret, news = cv2.threshold(new, 10, 255, cv2.THRESH_BINARY)
    
    
    itext = cv2.cvtColor(itext,cv2.COLOR_BGR2GRAY)
    tret, text = cv2.threshold(itext, 10, 255, cv2.THRESH_BINARY)
    news =  cv2.resize(news, (cols ,rows), interpolation=cv2.INTER_CUBIC)
    # img = cv2.bitwise_or(news,text)
    # img = cv2.addWeighted(text, 1 ,news, 1, 0)
    # cv2.imshow('nret',img)
    
    # timg = cv2.bitwise_and(img,img,mask = text)
    # cv2.imshow('img1_bg',timg)
    img1_bg = cv2.bitwise_and(img,img,mask = mask)
    img2_fg = cv2.bitwise_and(bcg,bcg,mask = mask_inv)
    # cv2.imshow('img1_bg',img1_bg)
    # cv2.imshow('img2_fg',img2_fg)
    img2_fg = cv2.dilate ( img2_fg , None , iterations = 3 ) 
 
    
    
    dst = cv2.addWeighted(img2_fg, 1 ,img1_bg, 1, 0)
    # dst = cv2.GaussianBlur ( dst ,  ( 9 ,  9 ) , 0 )
   
    
    # cv2.imshow('IG',dst)
    return dst
    # cv2.waitKey(0)

def main():
    
    detect(profile('15.jpg'))
    
 
if __name__ == '__main__':
    main()


