import numpy as np
import cv2
import os
import argparse
from ReadCameraModel import *

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
	f = f/f[2,2]
	# print(f)
	return f

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
		if m1.distance < 0.7 * m2.distance:
			im1_idx = m1.queryIdx
			im2_idx = m1.trainIdx
			[x1,y1] = kp1[im1_idx].pt
			[x2,y2] = kp2[im2_idx].pt
			img1_points.append([int(x1), int(y1)])
			img2_points.append([int(x2), int(y2)])
			good.append([m1])
	
	# orb = cv2.ORB_create(nfeatures=2000)
	# kp1 = orb.detect(img1, None)
	# kp1, des1 = orb.compute(img1, kp1)

	# kp2 = orb.detect(img2, None)
	# kp2, des2 = orb.compute(img2, kp2)

	# bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# matches=bf.match(des1,des2)
	# img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
	# img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
	# matches = sorted(matches, key = lambda x:x.distance)

	# for mat in matches[:50]:

	# 	# Get the matching keypoints for each of the images
	# 	img1_idx = mat.queryIdx
	# 	img2_idx = mat.trainIdx

	# 	# x - columns
	# 	# y - rows
	# 	# Get the coordinates
	# 	[x1,y1] = kp1[img1_idx].pt
	# 	[x2,y2] = kp2[img2_idx].pt

	# 	cv2.circle(img1, tuple([int(x1), int(y1)]), 10, (0, 255, 0))
	# 	cv2.circle(img2, tuple([int(x2), int(y2)]), 10, (0, 255, 0))
	# 	img1_points.append([int(x1), int(y1)])
	# 	img2_points.append([int(x2), int(y2)])

	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None)
	# cv2.imshow("img1", img3)
	# cv2.waitKey(0)

	return np.asarray(img1_points), np.asarray(img2_points)

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    cv2.imshow("img1", img1)
    cv2.imwrite("./report_images/removing_car.png", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    # return img1,img2

def find_essential_matrix(intr, F):
	E = np.matmul(intr.T,np.matmul(F,intr))
	U, S, Vh = np.linalg.svd(E)
	S[-1] = 0
	E = np.diag(S)
	E = np.matmul(U, np.matmul(S, Vh))

	return E
	# W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	# C = U[:,2]
	# R = np.matmul(U, np.matmul(W, Vh))
	# if np.linalg.det(R) > 0:
	# 	return R, C
	# else:
	# 	return -R, -C

def multiply_three(a, b, c):

	return np.matmul(a, np.matmul(b, c))

def estimate_camera_pose(E):

	poses = []
	U, S, Vh = np.linalg.svd(E, full_matrices = True)
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	U3 = U[:,2]
	U3 = np.reshape(U3, (3,1))

	C1 = U3.copy()
	R1 = multiply_three(U, W, Vh)

	if np.linalg.det(R1) < 0:
		R1 = -R1
		C1 = -C1

	poses.append(np.hstack((R1, C1)))

	C2 = -U3.copy()
	R2 = multiply_three(U, W, Vh)

	if np.linalg.det(R2) < 0:
		R2 = -R2
		C2 = -C2

	poses.append(np.hstack((R2, C2)))
	
	C3 = U3.copy()
	R3 = multiply_three(U, W.T, Vh)

	if np.linalg.det(R3) < 0:
		R3 = -R3
		C3 = -C3

	poses.append(np.hstack((R3, C3)))
	
	C4 = -U3.copy()
	R4 = multiply_three(U, W.T, Vh)

	if np.linalg.det(R4) < 0:
		R4 = -R4
		C4 = -C4

	poses.append(np.hstack((R4, C4)))

	return poses


def fundamental_matrix_ransac(img1_points, img2_points):

	max_iter = 4000
	threshold = 60
	diff = np.array([[643.788025, 484.40799]])
	pts1 = img1_points.copy()
	pts2 = img2_points.copy()
	ones = np.ones((img1_points.shape[0], 1))
	pts1 = np.hstack((pts1, ones))
	pts2 = np.hstack((pts2, ones))
	all_index = np.arange(0, img1_points.shape[0])
	min_error = 10000000
	best_F = []
	np.random.seed(40)

	for i in range(max_iter):

		index = np.random.choice(all_index, 8, replace=False)
		F = find_f(img1_points[index], img2_points[index])
		error = np.matmul(np.matmul(pts2, F), pts1.T)

		if np.abs(np.sum(error)) < min_error:
			min_error = np.abs(np.sum(error))
			best_F = F
			print(min_error)

		max_iter += 1

	return best_F


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--input", default = '/media/arjun/My Passport/Oxford_dataset/stereo/centre/', help = "Path of the images")
	parser.add_argument("--model", default = './model', help = "Path of the images")
	Flags = parser.parse_args()

	img1 = cv2.imread('./data/1399381497885112.png', 0)
	img2 = cv2.imread('./data/1399381498322587.png', 0)
	img1_feat = img1[:756,:]
	img2_feat = img2[:756,:]

	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(Flags.model)

	img1_points, img2_points = find_features(img1_feat, img2_feat)
	F = fundamental_matrix_ransac(img1_points, img2_points)

	F_, _ = cv2.findFundamentalMat(np.float32(img1_points), np.float32(img2_points), method=cv2.FM_RANSAC)
	print(F, F_)

	lines1 = cv2.computeCorrespondEpilines(img2_points.reshape(-1,1,2), 2, F)
	lines1 = lines1.reshape(-1,3)

	lines2 = cv2.computeCorrespondEpilines(img2_points.reshape(-1,1,2), 2, F_)
	lines2 = lines1.reshape(-1,3)
	drawlines(img1.copy(), img2.copy(), lines1, img1_points, img2_points)
	drawlines(img1, img2, lines2, img1_points, img2_points)
	intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
	# print(intrinsic)
	E = find_essential_matrix(intrinsic, F)
	poses = estimate_camera_pose(E)
