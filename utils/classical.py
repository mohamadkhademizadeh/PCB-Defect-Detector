import cv2, numpy as np

def detect_defects(test_bgr, ref_bgr=None, blur=3, thr=15, min_area=80):
    gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    if blur>0:
        gray = cv2.GaussianBlur(gray, (2*blur+1,2*blur+1), 0)
    if ref_bgr is not None:
        ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        if blur>0:
            ref = cv2.GaussianBlur(ref, (2*blur+1,2*blur+1), 0)
        diff = cv2.absdiff(gray, ref)
        _, binm = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    else:
        # heuristic: emphasize scratches/holes
        edges = cv2.Canny(gray, 50, 150)
        binm = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)
    contours,_ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area: 
            continue
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,x+w,y+h))
    return binm, boxes
