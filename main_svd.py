import numpy as np
import cv2
import os

def find_f(img1_pts, img2_pts):

	
	M = np.array([[img1_pts[0,0]*img2_pts[0,0], img1_pts[0,0]*img2_pts[0,1], img1_pts[0,0], img1_pts[0,1]*img2_pts[0,0], img1_pts[0,1]*img2_pts[0,1], img1_pts[0,1], \
				img2_pts[0,0], img2_pts[0,1], 1],\
				[img1_pts[1,0]*img2_pts[1,0], img1_pts[1,0]*img2_pts[1,1], img1_pts[1,0], img1_pts[1,1]*img2_pts[1,0], img1_pts[1,1]*img2_pts[1,1], img1_pts[1,1], \
				img2_pts[1,0], img2_pts[1,1], 1],\
				[img1_pts[2,0]*img2_pts[2,0], img1_pts[2,0]*img2_pts[2,1], img1_pts[2,0], img1_pts[2,1]*img2_pts[2,0], img1_pts[2,1]*img2_pts[2,1], img1_pts[2,1], \
				img2_pts[2,0], img2_pts[2,1], 1],\
				[img1_pts[3,0]*img2_pts[3,0], img1_pts[3,0]*img2_pts[3,1], img1_pts[3,0], img1_pts[3,1]*img2_pts[3,0], img1_pts[3,1]*img2_pts[3,1], img1_pts[3,1], \
				img2_pts[3,0], img2_pts[3,1], 1],\
				[img1_pts[4,0]*img2_pts[4,0], img1_pts[4,0]*img2_pts[4,1], img1_pts[4,0], img1_pts[4,1]*img2_pts[4,0], img1_pts[4,1]*img2_pts[4,1], img1_pts[4,1], \
				img2_pts[4,0], img2_pts[4,1], 1],\
				[img1_pts[5,0]*img2_pts[5,0], img1_pts[5,0]*img2_pts[5,1], img1_pts[5,0], img1_pts[5,1]*img2_pts[5,0], img1_pts[5,1]*img2_pts[5,1], img1_pts[5,1], \
				img2_pts[5,0], img2_pts[5,1], 1],\
				[img1_pts[6,0]*img2_pts[6,0], img1_pts[6,0]*img2_pts[6,1], img1_pts[6,0], img1_pts[6,1]*img2_pts[6,0], img1_pts[6,1]*img2_pts[6,1], img1_pts[6,1], \
				img2_pts[6,0], img2_pts[6,1], 1],\
				[img1_pts[7,0]*img2_pts[7,0], img1_pts[7,0]*img2_pts[7,1], img1_pts[7,0], img1_pts[7,1]*img2_pts[7,0], img1_pts[7,1]*img2_pts[7,1], img1_pts[7,1], \
				img2_pts[7,0], img2_pts[7,1], 1]])
	# print(M)			
	U, S, Vh = np.linalg.svd(M, full_matrices=True)
	# print(S)
	S[-1] = 0
	# M_ = np.
	S = np.diag(S)
	z = np.zeros((8,1))
	S = np.hstack((S,z))
	M_ = np.matmul(U,np.matmul(S,Vh))
	# print(M_)
	U, S, Vh = np.linalg.svd(M, full_matrices=True)
	# print(type(Vh))
	f = Vh[8,:]
	# print(f)
	f = f.reshape(3,3)
	# print(f)
	return f	

def find_essential_matrix(intr, F):
	E = np.matmul(intr.T,np.matmul(F,intr))
	U, S, Vh = np.linalg.svd(E)
	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	C = U(:,3)
	R = np.matmul(U, np.matmul(W, Vh))
	if np.linalg.det(R) > 0:
		return R, C
	else:
		return -R, -C	
	


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
	# for m in matches:
		
	# 	# print("mQ", m.queryIdx)
	# 	# print("nQ", n.queryIdx)
	# 	# print("mt", m.trainIdx)
	# 	# print("nt", n.trainIdx)
	# 	# exit(0)
	# 	# if m.distance < 0.2*n.distance:
	# 	# 	good.append([m])
	# 	im1_idx = m.queryIdx
	# 	im2_idx = m.trainIdx
	# 	(x1,y1) = kp1[im1_idx].pt
	# 	(x2,y2) = kp2[im2_idx].pt
	# 	print(x1,y1,x2,y2)	
	# 	exit(0)




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
		# eightpts = np.random.choice(key_matches, 8, replace=False)
		# print(eightpts)
		# exit(0)
		eightpts1 = np.array([[1,1], [3,1], [3,5], [7,5], [7,9], [11,9], [11,13], [15,13]])
		eightpts2 = np.array([[1.5,1.5], [3.5,1.5], [3.5,5.5], [10.5,15.5], [7.5,9.5], [11.5,9.5], [11.5,13.5], [19.5,0.5]])
		F = find_f(eightpts1, eightpts2)
		exit(0)
		
	fx, fy, cx, cy = 0.1, 0.2, 23, 25	
	intrinsic = np.array([[0.1, 0, 23], [0, 0.2, 25], [0, 0, 1]])
	E = find_essential_matrix(intrinsic, F)