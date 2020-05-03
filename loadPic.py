# -*- coding: utf-8 -*-
import oss2
from itertools import islice
import numpy as np
import cv2
import dlib
import os
import datetime
# 阿里云主账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM账号进行API访问或日常运维，请登录 https://ram.console.aliyun.com 创建RAM账号。
auth = oss2.Auth('LTAI4G5et2Fa3uZ7cmW4usY9', '8iAalIbDY0yEn8YbKUC1lkx42eaXZ5')
bucket = oss2.Bucket(auth, 'http://oss-cn-shanghai.aliyuncs.com', 'fdb-tagging')


def landMarks(path):
    # cv2读取图像
    img = cv2.imread(path)

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 1)
    if (len(rects)>1):
        create_str_to_txt(datetime.datetime.now().strftime('%Y-%m-%d'),path)
    else:
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                print(idx,pos)

                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(img, pos, 5, color=(0, 255, 0))
                # 利用cv2.putText输出1-68
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(idx+1), pos, font, 0.7, (0, 0, 255), 1,cv2.LINE_AA)
                # 9 下巴    34 鼻尖
                # if ((idx+1)==34):
                #     return pos
        
        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)


def loadPic(auth,bucket):
	for b in islice(oss2.ObjectIterator(bucket, prefix='image_raw/Male/2019_'), 4):
		keyName = b.key 
		if ('.jpg' in keyName):
			print('file: ' + keyName)
			bucket.get_object_to_file(keyName, '1.jpg')

		else:
		    print('folder')	

		pos = landMarks('1.jpg')
		print (pos)


if __name__ == '__main__':
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	auth = oss2.Auth('LTAI4G5et2Fa3uZ7cmW4usY9', '8iAalIbDY0yEn8YbKUC1lkx42eaXZ5')
	bucket = oss2.Bucket(auth, 'http://oss-cn-shanghai.aliyuncs.com', 'fdb-tagging')
	loadPic(auth,bucket)
	landmarks('1.jpg')