import cv2, numpy as np

def overlay_mask(img_bgr, mask):
    color = (0,0,255)
    out = img_bgr.copy()
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out

def draw_boxes(img_bgr, boxes):
    out = img_bgr.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    return out
