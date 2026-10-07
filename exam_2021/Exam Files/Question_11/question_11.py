import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

img = cv2.imread('books.png')
# Changing the order from bgr to rgb so that matplotlib can show it
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap=cm.gray)

plt.pause(0.3)

edges = cv2.Canny(gray, 100, 200)
plt.imshow(edges)

plt.pause(0.3)


dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
print(len(dst))


above_thres = 0

for x in dst:
    for y in x:
        if y > 0.01:
            above_thres +=1

print(above_thres)


above_thres = np.sum(dst > 0.01)

print("Total elements:", dst.size)
print("Values > 0.01:", above_thres)