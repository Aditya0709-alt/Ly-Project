import cv2
import numpy as np
import os

def image_mosaicing(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Brute-force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Estimate homography matrix using all good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, 0)

    # Warp the images
    img1_warped = cv2.warpPerspective(img1, homography_matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    # Combine the images
    img1_warped[:, :img2.shape[1]] = img2

    return img1_warped

# Example usage
folder_path = r'C:\Users\mbhar\Desktop\Chopy\Ly-Project\output'

# Load images
image1_path = os.path.join(folder_path, 'left_half_set2_1.tif')
image2_path = os.path.join(folder_path, 'right_half_set2_1.tif')

# Load images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

result = image_mosaicing(image1, image2)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
