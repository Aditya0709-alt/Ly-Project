import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

folder_path = r'C:\Users\mbhar\Desktop\Chopy\Ly-Project\output'

# Load images
image1_path = os.path.join(folder_path, 'left_half_set2_1.tif')
image2_path = os.path.join(folder_path, 'right_half_set2_1.tif')



# Load images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# # Initialize SIFT detector
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

homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

h, w = image1.shape[:2]
corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
warped_corners = cv2.perspectiveTransform(corners, homography_matrix)
min_x, min_y = np.int32(warped_corners.min(axis=0).ravel())
max_x, max_y = np.int32(warped_corners.max(axis=0).ravel())
width_of_result = max_x - min_x
height_of_result = max_y - min_y

# Adjust the homography matrix to shift the image properly
shift_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
final_homography_matrix = np.dot(shift_matrix, homography_matrix)


height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]

# Stack the images vertically to create a canvas for stitching
canvas = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)

# Warp image2 to align with image1 using the homography matrix
image2_warped = cv2.warpPerspective(image2, homography_matrix, (width1 + width2, max(height1, height2)))

# Copy image1 to the left side of the canvas
canvas[:height1, :width1, :] = image1

# Copy the warped image2 to the right side of the canvas
canvas[:height2, width1:, :] = image2_warped[:height2, :width2, :]

blend_width = 0  # Adjust this value based on the size of the overlapping region
blend_mask = np.zeros((max(height1, height2), blend_width))

# Create a linear gradient for the blend mask
for i in range(blend_width):
    blend_mask[:, i] = np.linspace(1, 0, blend_mask.shape[0])
canvas = canvas.astype(float)
# Apply the feathering mask to the overlapping region
canvas[:height1, width1 - blend_width:width1, :] *= (1 - blend_mask[:, :, np.newaxis])
canvas[:height2, width1 - blend_width:width1, :] += image2_warped[:height2, :blend_width, :] * blend_mask[:, :, np.newaxis]
canvas = canvas.astype(np.uint8)


# # Convert BGR to RGB for displaying with Matplotlib
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

plt.imshow(canvas_rgb)
plt.axis('off')
plt.show()

