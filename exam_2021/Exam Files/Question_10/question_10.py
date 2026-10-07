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

# Theta = the angular resolution for line orientation in the Hough Transform.

# rho = 1 means the Hough Transform tests line distances in steps of 1 pixel.
# Intuition - RHO
# Imagine sliding a ruler around the image at different angles.
# The theta sets the angle of the ruler.
# The rho sets how finely you shift the ruler perpendicular to itself.

# Smaller rho → more bins → more precise detection → more computation.
# Larger rho → fewer bins → less precise detection.

lines = cv2.HoughLines(edges, rho=1, theta=0.0017, threshold=200) 

print(len(lines))