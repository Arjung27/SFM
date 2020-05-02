import numpy as np
import cv2
import os

def find_f(pts):
	# M = 
	pass

def find_features(img1, img2):

	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	good = []
	img1_points = []
	img2_points = []
	for m1, m2 in matches:
		if m1.distance < 0.1 * m2.distance:
			im1_idx = m1.queryIdx
			im2_idx = m1.trainIdx
			(x1,y1) = kp1[im1_idx].pt
			(x2,y2) = kp2[im2_idx].pt
			img1_points.append((x1, y1))
			img2_points.append((x2, y1))
			good.append([m1])

	return img1_points, img2_points, good

def fundamental_matrix_ransac(img1_points, img1_points):

	max_iter = 400
	threshold = 60

	for i in range(max_iter):

		

if __name__ == '__main__':

	img1 = cv2.imread('./data/1399381446267196.png', 0)
	img2 = cv2.imread('./data/1399381446329684.png', 0)

	img1_points, img2_points, good = find_features(img1, img2)
	F = fundamental_matrix_ransac(img1_points, img2_points)