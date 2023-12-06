import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

folder_path = r'C:\Users\mbhar\Desktop\Chopy\Ly-Project\output'

# Load images
image1_path = os.path.join(folder_path, 'set6_1.tif')
image2_path = os.path.join(folder_path, 'set6_2.tif')



# Load images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Use SIFT to find keypoints and descriptors
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Use a FLANN based Matcher to find matches
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to get good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract corresponding points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find the perspective transformation matrix
M, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# Get the dimensions of the first image
h1, w1 = image1.shape[:2]

# Warp the second image to align with the first
warped_image2 = cv2.warpPerspective(image2, M, (w1 + image2.shape[1], h1))

# Combine the two images
result = np.copy(warped_image2)
result[:h1, :w1] = image1

canvas_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
# Display the result
plt.imshow(canvas_rgb)
plt.axis('off')
plt.show()
