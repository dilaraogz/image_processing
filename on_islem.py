import cv2

#fonksiyon olusturma
#görüntü boyutlandırma
def boyutlandırma(img,x,y):
    return cv2.resize(img,(x,y))
