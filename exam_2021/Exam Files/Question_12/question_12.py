import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

img = cv2.imread('things1.png')
img2 = cv2.imread('things2.png')
# Changing the order from bgr to rgb so that matplotlib can show it
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])

b,g,r = cv2.split(img2)
img2 = cv2.merge([r,g,b])

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)


feat1 = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
feat2, status, error = cv2.calcOpticalFlowPyrLK(gray, gray2, feat1, None)

max_dx = 0

for i in range(len(feat2)):
    x1 = feat1[i][0][0]
    x2 = feat2[i][0][0]
    
    dx = abs(x2 - x1)
    if dx > max_dx:
        max_dx = dx

print("Max horizontal displacement:", max_dx)