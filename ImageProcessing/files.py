import cv2
import base64
from os.path import dirname, abspath, join

BASE_DIR = dirname(dirname(abspath(__file__)))

# 素材
MATERIAL = {
    'comic': join(BASE_DIR, 'Material', 'special_effect_box.png'),
    'gogogo': join(BASE_DIR, 'Material', 'gogogo.png'),
}

# 分級器
CASCADE_FILE = {
    'frontalface':
    join(BASE_DIR, 'Classifier', 'haarcascade_frontalface_default.xml'),
    'frontalface_alt2':
    join(BASE_DIR, 'Classifier', 'haarcascade_frontalface_alt2.xml'),
    'frontalface_alt_tree':
    join(BASE_DIR, 'Classifier', 'haarcascade_frontalface_alt_tree.xml'),
    'fullbody':
    join(BASE_DIR, 'Classifier', 'haarcascade_fullbody.xml'),
    'upperbody':
    join(BASE_DIR, 'Classifier', 'haarcascade_upperbody.xml'),
    'mcs_upperbody':
    join(BASE_DIR, 'Classifier', 'haarcascade_mcs_upperbody.xml'),
}

# 測試圖片
TEST_IMAGE = {
    '0': join(BASE_DIR, 'TestImage', '0.jpg'),
    '1': join(BASE_DIR, 'TestImage', '1.jpg'),
    '2': join(BASE_DIR, 'TestImage', '2.jpg'),
    '3': join(BASE_DIR, 'TestImage', '3.jpg'),
    '4': join(BASE_DIR, 'TestImage', '4.jpg'),
    '5': join(BASE_DIR, 'TestImage', '5.jpg'),
    '6': join(BASE_DIR, 'TestImage', '6.jpg'),
    '7': join(BASE_DIR, 'TestImage', '7.jpg'),
    '8': join(BASE_DIR, 'TestImage', '8.jpg'),
    '9': join(BASE_DIR, 'TestImage', '9.jpg'),
    '10': join(BASE_DIR, 'TestImage', '10.jpg'),
    '11': join(BASE_DIR, 'TestImage', '11.jpg'),
    '12': join(BASE_DIR, 'TestImage', '12.jpg'),
    '13': join(BASE_DIR, 'TestImage', '13.jpg'),
    '14': join(BASE_DIR, 'TestImage', '14.jpg'),
    '15': join(BASE_DIR, 'TestImage', '15.jpg'),
    '16': join(BASE_DIR, 'TestImage', '16.jpg'),
    '17': join(BASE_DIR, 'TestImage', '17.jpg'),
    '18': join(BASE_DIR, 'TestImage', '18.jpg'),
    '19': join(BASE_DIR, 'TestImage', '19.jpg'),
    '20': join(BASE_DIR, 'TestImage', '20.jpg'),
    '21': join(BASE_DIR, 'TestImage', '21.jpg'),
    '22': join(BASE_DIR, 'TestImage', '22.jpg'),
    '23': join(BASE_DIR, 'TestImage', '23.jpg'),
    '24': join(BASE_DIR, 'TestImage', '24.jpg'),
    '25': join(BASE_DIR, 'TestImage', '25.jpg'),
}

IMAGES = []


def imShow(cv_image):
    base64_str = cv2.imencode('.jpg', cv_image)[1].tostring()
    base64_str = base64.b64encode(base64_str).decode('utf-8')

    IMAGES.append(base64_str)
