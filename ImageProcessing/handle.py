"""
影像處理主體
"""
import imghdr, cv2, base64
from .draw import draw
from .matting import matting
from .files import MATERIAL, CASCADE_FILE


def handle(fileName):
    if not imghdr.what(fileName):
        raise RuntimeError("Could not read file: %s" % fileName)

    image = cv2.imread(fileName)

    # 拆出前景背景
    mask = matting(image)
    output = draw(image, mask)

    return output
