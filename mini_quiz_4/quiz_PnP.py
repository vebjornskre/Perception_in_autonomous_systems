import cv2
import numpy as np

# These are the result of going from world to camera
rvec = np.array([-0.05, -1.51, -0.00])
tvec = np.array([87.39, -2.25, -24.89])

# Point in camera coordinates
X_cam = [-6.71, 0.23, 21.59]

R, jacobian = cv2.Rodrigues(rvec)

# We are given P in camera coordinates, want it in world coordinates
# So we inverse the transfomration to go the other way. We can do it because
# inverse of rotation matrix is its transpose and inverse of translation is just
# X_cam - translation

X_world = np.transpose(R) @ (X_cam - tvec)

print(X_world)