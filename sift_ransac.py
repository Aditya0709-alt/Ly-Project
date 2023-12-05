import cv2
import numpy as np
import os


folder_path = r'C:\Users\mbhar\Desktop\Chopy\Ly-Project\output'

# Load images
image1_path = os.path.join(folder_path, 'left_half_set2_1.tif')
image2_path = os.path.join(folder_path, 'right_half_set2_1.tif')

# Load images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Use a FLANN based matcher
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

# Perform the k-nearest neighbor matching
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract the matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Use RANSAC to estimate the homography matrix
homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the first image onto the second
result = cv2.warpPerspective(image1, homography_matrix, (image2.shape[1] + image1.shape[1], image2.shape[0]))

# Copy the second image onto the result image
result[0:image2.shape[0], 0:image2.shape[1]] = image2


# Display the result
cv2.imshow('Mosaic', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

