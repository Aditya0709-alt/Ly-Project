import cv2
import numpy as np

# Load the two images
image1 = cv2.imread(
    "D:\College Material\LY COURSES\LY  PROJECT\Ly-Project\output\left_half_set2_1.tif",
    cv2.IMREAD_GRAYSCALE,
)
image2 = cv2.imread(
    "D:\College Material\LY COURSES\LY  PROJECT\Ly-Project\output\right_half_set2_1.tif",
    cv2.IMREAD_GRAYSCALE,
)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Use the BFMatcher to find the best matches between the descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# Display the matches
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract matched key points
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
    -1, 1, 2
)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
    -1, 1, 2
)

# Find Homography matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Use the Homography matrix to warp image2 and stitch it onto image1
result = cv2.warpPerspective(
    image2, M, (image1.shape[1] + image2.shape[1], image2.shape[0])
)
result[: image1.shape[0], : image1.shape[1]] = image1

# Display the stitched image
cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
