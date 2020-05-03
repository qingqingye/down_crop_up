# from imutils.object_detection import non_max_suppression
# import numpy as np
# import imutils
# import cv2
 
 
# # initialize the HOG descriptor/person detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 
# # load the image and resize it to (1) reduce detection time
# # and (2) improve detection accuracy
# image = cv2.imread('2019_AW_1.jpg')
# image = imutils.resize(image, width=min(400, image.shape[1]))
# orig = image.copy()

# # detect people in the image
# (rects, weights) = hog.detectMultiScale(
# image, winStride=(4, 4), padding=(8, 8), scale=1.05
# )

# # draw the original bounding boxes
# # for (x, y, w, h) in rects:
# #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

# #     # apply non-maxima suppression to the bounding boxes using a
# #     # fairly large overlap threshold to try to maintain overlapping
# #     # boxes that are still people
# #     rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])


# pick = non_max_suppression(rects, probs=1, overlapThresh=0.15)
# # draw the final bounding boxes
# for (xA, yA, xB, yB) in pick:
#     cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

# # show the output images
# # cv2.imshow("Before NMS", orig)
# cv2.imshow("After NMS", image)
# cv2.waitKey()


# import cv2

# def detect(filename):
#     face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    
#     img = cv2.imread(filename)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
#     faces = face_cascade.detectMultiScale(gray,1.3,5)
    
#     for (x,y,w,h) in faces:
#         img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    
#     cv2.namedWindow('Person Detected!')
#     cv2.imshow('Person Detected!',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     detect('2019_AW.jpg')


#!/usr/bin/python3
# -*- coding: utf-8 -*-
# import cv2
# import sys,os
# class opencvs():

#     def main(self):
     
#         #告诉OpenCV使用什么识别分类器
#         classfier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

#         frame = cv2.imread('test2.jpg')

#         #将当前帧转换成灰度图像
#         grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         #检测结果
#         faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 4, minSize = (50, 150))

#         #第一个参数是灰度图像
#         #第三个参数是人脸检测次数，设置越高，误检率越低，但是对于迷糊图片，我们设置越高，越不易检测出来

#         print( len(faceRects) )
#         for faceRect in faceRects:
#             x, y, w, h = faceRect
#             cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0) , 1)
            



#         #显示图像
#         cv2.imshow(' ', frame)           
#         cv2.waitKey()


# if __name__ == '__main__':
#     opencvs().main()

import numpy as np
import cv2 as cv
import tkinter

def compress(img):
    global oX,oY
    x, y = img.shape[0:2]
    oX = x
    oY = y
    # 缩放到原来的二分之一，输出尺寸格式为（宽，高）
    newimage = cv.resize(img, (int(y / 1.5), int(x / 1.5)))
    return newimage 

def relarge(img):
    global oX,oY
    x, y = img.shape[0:2]
    # 缩放到原来的二分之一，输出尺寸格式为（宽，高）
    newimage = cv.resize(img, (oY, oX))
    return newimage 

body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')
upper_cascade = cv.CascadeClassifier('haarcascade_upperbody.xml')
lower_cascade = cv.CascadeClassifier('haarcascade_lowerbody.xml')
img = cv.imread('test2.jpg')
img = compress(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

bodies = body_cascade.detectMultiScale(gray, 1.3, 1)
upper = upper_cascade.detectMultiScale(gray, 1.3, 1)
lower = lower_cascade.detectMultiScale(gray, 1.3, 1)

print(len(bodies),len(upper),len(lower))
for (x,y,w,h) in bodies:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]


for (x,y,w,h) in upper:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

for (x,y,w,h) in lower:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()