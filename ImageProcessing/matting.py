"""
拆出前景背景
"""
import cv2, numpy as np
from ImageProcessing.alpha_matting import closed_form, solve_fg_bg
from ImageProcessing.files import CASCADE_FILE, get_test_image, im_show

BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10


def matting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 邊緣偵測
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # 用邊緣取得輪廓
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # 製作遮罩
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # 遮罩修復
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)

    # # Closed Form Matting
    # trimap = cv2.imread(get_test_image('4-trimap'))
    # alpha = closed_form.with_scribbles(image / 255.0, trimap / 255.0)
    # foreground, background = solve_fg_bg(image, alpha)
    # im_show(alpha * 255.0)
    # im_show(foreground)
    # im_show(background)

    return mask.astype('uint8')
