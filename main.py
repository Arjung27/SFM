import numpy as np
import cv2
import os

def find_f(pts):
	# M = 
	pass


if __name__ == '__main__':

	img1 = cv2.imread('./data/1399381446267196.png', 0)
	img2 = cv2.imread('./data/1399381446329684.png', 0)

	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	bf = cv2.BFMatcher(cv2.NORM_L2)
	matches = bf.match(des1,des2)
	good = []
	# print(matches[1][0].queryIdx)
	# print(matches)
	for m in matches:
		
		# print("mQ", m.queryIdx)
		# print("nQ", n.queryIdx)
		# print("mt", m.trainIdx)
		# print("nt", n.trainIdx)
		# exit(0)
		# if m.distance < 0.2*n.distance:
		# 	good.append([m])
		im1_idx = m.queryIdx
		im2_idx = m.trainIdx
		(x1,y1) = kp1[im1_idx].pt
		(x2,y2) = kp2[im2_idx].pt
		print(x1,y1,x2,y2)	
		exit(0)




	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[0:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	# cv2.imshow("Image", img3)
	# cv2.waitKey(0)
	key_matches = good
	# qidx = key_matches.queryIdx
	# tidx = key_matches.trainIdx
	# print(qidx)
	match_found = False
	np.random.seed(0)
	while(not match_found):
		eightpts = np.random.choice(key_matches, 8, replace=False)
		print(eightpts)
		exit(0)
		# F = find_f(eightpts)
		print
