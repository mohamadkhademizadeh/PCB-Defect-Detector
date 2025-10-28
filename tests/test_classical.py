import numpy as np, cv2
from utils.classical import detect_defects

def test_basic():
    img = np.zeros((200,200,3), np.uint8)
    cv2.line(img, (10,10),(190,10),(255,255,255),2)
    m, boxes = detect_defects(img, None)
    assert m is not None
