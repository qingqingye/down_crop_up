import base64
import io
import os
import cv2
import numpy as np
import requests
from PIL import Image


class ImageReader:
    def read(self, path, return_type='byte+rgb'):
        mode_list = return_type.split('+')
        if 'byte' in mode_list:
            return open(path, 'rb').read()
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'rgb' in mode_list:
            img = img[:, :, ::-1]
        return img

    def create_str_to_txt(self, fileName, data):
        """
        创建txt，并且写入
        """
        # path_file_name = './action_{}.txt'.format(date)
        path_file_name = fileName + '.txt'
        if not os.path.exists(path_file_name):
            with  open(path_file_name, "w", encoding='utf-8') as f:
                print(f)

        with open(path_file_name, "a", encoding='utf-8') as f:
            f.write(data + '\r\n')


class RequestAgent:
    def __init__(self, url, key, secret):
        self.url = url
        self.header = {'api_key': key,
                       'api_secret': secret}
        self.data = {'image_file': None}

    def get_result(self, data, comple=None):
        self.data['image_file'] = data
        if comple is not None:
            self.header.update(comple)
        return requests.post(self.url, data=self.header, files=self.data)


class Crop:
    def nose2feet(img_origin, use_nose_point, use_feet_point):
        ratio = 1065 / 2048  # width/lenght
        length = use_feet_point[1] - use_nose_point[1]
        width = ratio * length
        # print(use_nose_point[0],use_feet_point[1])
        img = img_origin.crop(((use_nose_point[0] - (width / 2)), use_nose_point[1], (use_nose_point[0] + (width / 2)),
                               use_feet_point[1]))  # x1:x2 ; y1:y2
        return img


if __name__ == '__main__':
    img_root = '2020'
    reader = ImageReader()  # reader is the class

    # body_agent_url = 'https://api-cn.faceplusplus.com/humanbodypp/v2/segment'
    # face_agent_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    # api_key = 'hzitXbntiIf8ZSN7rxeLbyNmg1IgUBPB'
    # api_secret = '8Wq94Gh8rbN7BaEPdYuRuWEKf6rPhtkF'

    body_agent = RequestAgent(body_agent_url, api_key, api_secret)
    face_agent = RequestAgent(face_agent_url, api_key, api_secret)
    for root, dirs, files in os.walk(img_root):
        for fname in files:
            # print(os.path.join(root, fname))
            picPath = os.path.join(root, fname)
            # if fname.endswith('.png') or  fname.endswith('.jpg') or fname == '.DS_Store':
            #     continue
            img = reader.read(picPath, 'byte+rgb')

            try:
                nose_point = face_agent.get_result(img, comple={'return_landmark': 1}).json()['faces']
                nose_point = nose_point[0]['landmark']['nose_contour_lower_middle']
                segment = base64.b64decode(body_agent.get_result(img).json()['result'])
                imggray = np.array(Image.open(io.BytesIO(segment)))
            except Exception as e:
                print('failed', fname, end='')
                print(e)
                reader.create_str_to_txt('fail_log', picPath)
                continue
            thred = 40
            imggray[imggray < thred] = 0  # background
            imggray[imggray >= thred] = 255  # body
            contours, hierarchy = cv2.findContours(imggray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find  max  contour
            max_cnt = contours[0].shape[0]
            max_contour = contours[0]
            for cnt in contours:
                if cnt.shape[0] > max_cnt:
                    max_cnt = cnt.shape[0]
                    max_contour = cnt
            # print(max_contour.shape)
            # find shoe boundary.
            indx, indy = (np.argmax(max_contour, axis=0).squeeze())
            point = max_contour[indy, :, :].squeeze().tolist()
            imggray = cv2.circle(imggray, tuple(point), radius=10, color=128, thickness=-1)
            imggray = cv2.circle(imggray, (nose_point['x'], nose_point['y']), radius=10, color=128,
                                 thickness=-1)

            use_nose_point = [nose_point["x"], nose_point["y"]]
            use_feet_point = [point[0], point[1]]  # [x,y]
            print(use_feet_point, use_nose_point)

            print('success', fname)
            img_origin = Image.open(os.path.join(root, fname))
            img_crop = Crop.nose2feet(img_origin, use_nose_point, use_feet_point)

            img_crop.save(os.path.join('crop', fname))
    # cv2.imwrite(os.path.join(img_root.replace('raw', 'final_result'),fname), img_crop)
    # cv2.imwrite(os.path.join(img_root.replace('raw', 'result'),
    #                          fname.replace('.jpg', '.png')), imggray)
