import cv2
import os


TARGET_PATH = './ellipse.png'
COMP_DIR = './image'


target_img = cv2.imread(TARGET_PATH, cv2.IMREAD_GRAYSCALE)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
detector = cv2.AKAZE_create()
(target_kp, target_des) = detector.detectAndCompute(target_img, None)

files = os.listdir(COMP_DIR)
for file in files:
    comp_path = COMP_DIR + '/' + file
    comparing_img = cv2.imread(comp_path, cv2.IMREAD_GRAYSCALE)
    (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
    matches = bf.match(target_des, comparing_des)
    dist = [m.distance for m in matches]
    ret = sum(dist) / len(dist)
    print(file, ret)
