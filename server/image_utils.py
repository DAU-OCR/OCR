import os

import cv2
import numpy as np


def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def warp_perspective(image, corners, output_size=(200, 60)):
    dst = np.array([
        [0, 0],
        [output_size[0]-1, 0],
        [output_size[0]-1, output_size[1]-1],
        [0, output_size[1]-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, output_size)

def get_plate_corners(image, fname=None, save_debug=False, debug_dir=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    if save_debug and debug_dir:
        cv2.imwrite(os.path.join(debug_dir, 'gray.png'), gray)
        cv2.imwrite(os.path.join(debug_dir, 'blur.png'), blur)
        cv2.imwrite(os.path.join(debug_dir, 'canny.png'), edged)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_vis = image.copy()
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
    if save_debug and debug_dir:
        cv2.imwrite(os.path.join(debug_dir, 'contours.png'), contour_vis)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            x, y, w, h = cv2.boundingRect(pts)
            if w / h < 2:
                continue
            return order_corners(pts)
    return None


def get_plate_corners_threshold(image, fname=None, save_debug=False, debug_dir=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

    if save_debug and debug_dir:
        cv2.imwrite(os.path.join(debug_dir, 'blur_thresh.png'), blur)
        cv2.imwrite(os.path.join(debug_dir, 'thresh.png'), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_vis = image.copy()
    cv2.drawContours(contour_vis, contours, -1, (0, 0, 255), 1)
    if save_debug and debug_dir:
        cv2.imwrite(os.path.join(debug_dir, 'contours_thresh.png'), contour_vis)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        ordered = order_corners(box.astype(np.float32))
        if save_debug and debug_dir:
            box_img = image.copy()
            cv2.polylines(box_img, [np.int32(ordered)], True, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, 'minAreaRect_box.png'), box_img)
        return ordered
    return None
