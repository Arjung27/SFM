import numpy as np
import cv2
import os

if __name__ == '__main__':

	img1 = cv2.imread('./data/1399381446267196.png', 0)
	img2 = cv2.imread('./data/1399381446329684.png', 0)

	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)
	good = []

	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])

	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv2.imshow("Image", img3)
	cv2.waitKey(0)
