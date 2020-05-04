# import cv2
# import numpy as np
# import dlib




# #read pic
# def readpic():
#     img = cv2.imdecode(np.fromfile('test.jpg', dtype=np.uint8),-1)
#     return img


# # detect face
# def detectFace():
#     img = cv2.imread("test.jpg")
#     detector = dlib.get_frontal_face_detector()
#     faces = detector(img,1)
#     if(len(faces)>0):
#         # not one face this pic is not ok
#         create_str_to_txt(datetime.datetime.now().strftime('%Y-%m-%d'),'filename')
#     else:
#         cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),
#             (255,0,0))
#         # k is the number of face, start from 0;  
#         # d is the coordinate of face lefttop&right bottom 
#         cv2.imshow("original",img)
#         cv2.waitKey(0) 
        

# if __name__ == '__main__':
#    # create_str_to_txt(  datetime.datetime.now().strftime('%Y-%m-%d'),'123')
#    #readpic();
#    detectFace();



import numpy as np
import cv2
import dlib
import os
import datetime


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



def create_str_to_txt(date,str_data):
    """
    创建txt，并且写入
    """
    path_file_name = './action_{}.txt'.format(date)
    if not os.path.exists(path_file_name):
        with open(path_file_name, "w") as f:
            print(f)

    with open(path_file_name, "a") as f:
        f.write(str_data)


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
                if ((idx+1)==34):
                    return pos
        
        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)



if __name__ == '__main__':
    pos = landMarks('2020_AW_Lisbon_Female_Aleksandar Protic_14.jpg')
    print(pos)
