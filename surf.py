import os
import cv2
import numpy as np


def image_mosaicing_surf(img1, img2):
    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create()

    # Find the keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Estimate homography matrix using all good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Warp the images
    img1_warped = cv2.warpPerspective(
        img1, homography_matrix, (img1.shape[1] + img2.shape[1], img1.shape[0])
    )

    # Combine the images
    img1_warped[:, : img2.shape[1]] = img2

    return img1_warped


# Example usage
folder_path = "D:\College Material\LY COURSES\LY  PROJECT\Ly-Project\output"

# Load images
image1_path = os.path.join(folder_path, "left_half_set2_1.tif")
image2_path = os.path.join(folder_path, "right_half_set2_1.tif")

# Load images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

result = image_mosaicing_surf(image1, image2)

cv2.imshow(result)
cv2.waitKey(0)
cv2.destroyAllWindows()
