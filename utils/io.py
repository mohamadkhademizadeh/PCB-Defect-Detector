import cv2, numpy as np

def read_bgr(path): 
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: 
        raise FileNotFoundError(path)
    return img

def align_to_ref(ref_bgr, img_bgr):
    # ORB feature-based homography alignment
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(ref_bgr, None)
    kp2, des2 = orb.detectAndCompute(img_bgr, None)
    if des1 is None or des2 is None:
        return img_bgr
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10: 
        return img_bgr
    src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: 
        return img_bgr
    aligned = cv2.warpPerspective(img_bgr, H, (ref_bgr.shape[1], ref_bgr.shape[0]))
    return aligned
