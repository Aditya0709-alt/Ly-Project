import cv2
import imutils

imageA = cv2.imread('output/left_half_set2_1.tif')
imageB = cv2.imread('output/right_half_set2_1.tif')

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kpsA, featuresA = sift.detectAndCompute(grayA, None)
kpsB, featuresB = sift.detectAndCompute(grayB, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(featuresA, featuresB, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

result = cv2.drawMatches(imageA, kpsA, imageB, kpsB, good_matches, None)

cv2.imshow("SIFT Feature Matching", imutils.resize(result, height=500))
cv2.waitKey(0)
