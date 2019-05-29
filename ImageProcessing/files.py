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

IMAGES = []


# 測試圖片
def get_test_image(fileName, ext='jpg'):
    return join(BASE_DIR, 'TestImage', '%s.%s' % (fileName, ext))


# 增加要顯示的圖片
def im_show(cv_image):
    base64_str = cv2.imencode('.jpg', cv_image)[1].tostring()
    base64_str = base64.b64encode(base64_str).decode('utf-8')

    IMAGES.append(base64_str)
