import numpy as np
import cv2
import os
import argparse
import glob
import math
import matplotlib.pyplot as plt
from ReadCameraModel import *
from UndistortImage import *

def rotationMatrixToEulerAngles(R) :
 
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
 
	return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

def multiply_three(a, b, c):

	return np.matmul(a, np.matmul(b, c))

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
	
	U, S, Vh = np.linalg.svd(M, full_matrices=True)
	F = np.reshape(Vh[-1, :], (3,3)).T
	U, S, Vh = np.linalg.svd(F)
	S[-1] = 0
	S = np.diag(S)
	f = multiply_three(U, S, Vh)
	# f = f/f[2,2]

	return f

def find_features_orb(img1, img2):

	orb = cv2.ORB_create(nfeatures=2000)
	kp1 = orb.detect(img1, None)
	kp1, des1 = orb.compute(img1, kp1)

	kp2 = orb.detect(img2, None)
	kp2, des2 = orb.compute(img2, kp2)

	bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches=bf.match(des1,des2)
	img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
	matches = sorted(matches, key = lambda x:x.distance)

	for mat in matches[:50]:

		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		# x - columns
		# y - rows
		# Get the coordinates
		[x1,y1] = kp1[img1_idx].pt
		[x2,y2] = kp2[img2_idx].pt

		cv2.circle(img1, tuple([int(x1), int(y1)]), 10, (0, 255, 0))
		cv2.circle(img2, tuple([int(x2), int(y2)]), 10, (0, 255, 0))
		img1_points.append([int(x1), int(y1)])
		img2_points.append([int(x2), int(y2)])

	return np.asarray(img1_points), np.asarray(img2_points)

def find_features(img1, img2):

	sift = cv2.xfeatures2d.SIFT_create() 

    # find the keypoints and descriptors with SIFT in current as well as next frame
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	
	features1 = [] # Variable for storing all the required features from the current frame
	features2 = [] # Variable for storing all the required features from the next frame

	# Ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.5*n.distance:
			features1.append(kp1[m.queryIdx].pt)
			features2.append(kp2[m.trainIdx].pt)

	return np.asarray(features1), np.asarray(features2)

def drawlines(img1, img2, lines, pts1, pts2, index):
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
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    return img1,img2

def find_essential_matrix(intr, F):

	E = np.matmul(intr.T,np.matmul(F,intr))
	U, S, Vh = np.linalg.svd(E)
	S[0] = 1
	S[1] = 1
	S[2] = 0
	S = np.diag(S)
	E = np.matmul(U, np.matmul(S, Vh))

	return E

def homogenous_matrix(mat):

	T = mat[:,-1, np.newaxis]
	R = mat[:, :3]
	# T = np.matmul(R, T)
	mat = np.hstack((R, T))

	return mat

def disambiguate_camera_pose(poses, world_points):

	count = np.zeros((4,1))

	for j in range(len(poses)):

		pose = poses[j]
		angles = rotationMatrixToEulerAngles(pose[:,:3])
		if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:
			points = world_points[:3, :, j]
			r3 = pose[2, :3, np.newaxis]
			C = np.reshape(pose[:, 3], (3,1))
			result = np.matmul(r3.T, (points - C))
			inds = np.where(result > 0)
			count[j] = len(inds[0])

	if np.max(count) > 0:
		return np.argmax(count)
	else:
		return -1


def skew(vec):

	return np.array([[0, -vec[2], vec[1]],
					 [vec[2], 0, -vec[0]],
					 [-vec[1], vec[0], 0]], dtype=np.float32)

def linear_triangulation(world, intrinsic, poses, pts1, pts2):

	world_points = np.zeros((4, pts1.shape[0], len(poses)))

	for j in range(len(poses)):

		pose = homogenous_matrix(poses[j])
		# pose = np.matmul(intrinsic, pose)

		for i in range(pts1.shape[0]):

				x1 = pts1[i][0]
				y1 = pts1[i][1]
				x2 = pts2[i][0]
				y2 = pts2[i][1]

				# http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
				p1 = skew(np.array([[x1], [y1], [1]]))
				p2 = skew(np.array([[x2], [y2], [1]]))
				pose1 = np.matmul(p1, world)
				pose2 = np.matmul(p2, pose)
				M = np.vstack((pose1, pose2))
				# M = np.vstack((world, pose))
				U, S, Vh = np.linalg.svd(M, full_matrices = True)
				world_points[:, i, j] = Vh[-1]/Vh[-1][3]

	return world_points

def estimate_camera_pose(E):

	poses = []
	U, S, Vh = np.linalg.svd(E, full_matrices = True)
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	U3 = U[:,2]
	U3 = np.reshape(U3, (3,1))

	C1 = U3.copy()
	# print(U, Vh)
	R1 = U @ W @ Vh #multiply_three(U, W, Vh)
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

	max_iter = 50
	threshold = 0.01
	min_inliers = 0
	pts1 = img1_points.copy()
	pts2 = img2_points.copy()
	ones = np.ones((img1_points.shape[0], 1))
	pts1 = np.hstack((pts1, ones))
	pts2 = np.hstack((pts2, ones))
	all_index = np.arange(0, img1_points.shape[0])
	min_error = 10000000
	best_F = []
	np.random.seed(40)

	# while (min_inliers/img1_points.shape[0]) < 0.2:
	for i in range(max_iter):

		index = np.random.choice(all_index, 8, replace=False)
		F = find_f(img1_points[index], img2_points[index])
		error = np.abs(np.diag(np.matmul(np.matmul(pts2, F), pts1.T)))
		inds = np.where(error < threshold)

		if len(inds[0]) > min_inliers:
			min_inliers = len(inds[0])
			best_F = F
			inliers1 = img1_points[inds[0]]
			inliers2 = img2_points[inds[0]]
			# print(min_inliers/img1_points.shape[0])
			

		# if min_inliers/img1_points.shape[0] > 0.99:
		# 	break
	# print(min_inliers/img1_points.shape[0])

	return best_F, inliers1, inliers2


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--input", default = '/cmlscratch/arjgpt27/projects/Oxford_dataset/stereo/centre/', help = "Path of the images")
	parser.add_argument("--model", default = './model', help = "Path of the images")
	Flags = parser.parse_args()

	prev_pose = np.array([[1, 0, 0, 0],
				  [0, 1, 0, 0],
				  [0, 0, 1, 0]], dtype = np.float32)

	init_world = np.array([[1, 0, 0, 0],
				  [0, 1, 0, 0],
				  [0, 0, 1, 0]], dtype = np.float32)

	files = np.sort(glob.glob(os.path.join(Flags.input, '*png'), recursive=True))
	fig = plt.figure()
	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(Flags.model)

	for i in range(19, len(files) - 1):

		print("Reading Frame ",i)
		img1 = cv2.imread(files[i], 0)
		color_image = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
		undistorted_image1 = UndistortImage(color_image, LUT)
		img1 = cv2.cvtColor(undistorted_image1, cv2.COLOR_BGR2GRAY)

		img2 = cv2.imread(files[i+1], 0)
		color_image = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
		undistorted_image2 = UndistortImage(color_image, LUT)
		img2 = cv2.cvtColor(undistorted_image2, cv2.COLOR_BGR2GRAY)

		img1_feat = img1[200:650,:]
		img2_feat = img2[200:650,:]

		fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(Flags.model)

		img1_points, img2_points = find_features(img1_feat, img2_feat)
		print(img1_points.shape[0])
		F, inliers1, inliers2 = fundamental_matrix_ransac(img1_points, img2_points)
		img1_points = inliers1
		img2_points = inliers2
		print(img1_points.shape[0])
		# print("Our calculation F: ", F)
		F_, _ = cv2.findFundamentalMat(np.float32(img1_points), np.float32(img2_points), method=cv2.FM_RANSAC)
		# print("Our calculation F: ", F)
		# print("CV Calculation F_: ", F_)
		# exit(-1)

		# lines1 = cv2.computeCorrespondEpilines(img2_points.reshape(-1,1,2), 2, F)

		# lines1 = lines1.reshape(-1,3)

		# lines2 = cv2.computeCorrespondEpilines(img2_points.reshape(-1,1,2), 2, F_)
		# lines2 = lines1.reshape(-1,3)
		# drawlines(img1.copy(), img2.copy(), lines1, img1_points, img2_points, '1')
		# drawlines(img1, img2, lines2, img1_points, img2_points, '3')
		# continue

		intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
		# print(intrinsic)
		E = find_essential_matrix(intrinsic, F)
		# print("Our essential E: ", E)
		poses = estimate_camera_pose(E)
		world_points = linear_triangulation(init_world, intrinsic, poses, img1_points, img2_points)
		camera_pose_idx = disambiguate_camera_pose(poses, world_points)

		old_x, old_y, old_z = prev_pose[:,3]

		if camera_pose_idx >= 0:

			###############################################################
			E_cv, mask = cv2.findEssentialMat(img1_points, img2_points, focal=fx, pp=(cx, cy), method=cv2.RANSAC)
			_,R,t,_ = cv2.recoverPose(E_cv, img1_points, img2_points, intrinsic)
			# current_pose = np.hstack((R, t))
			###############################################################

			current_pose = poses[camera_pose_idx]
			if current_pose[2, 3] > 0:
				current_pose[2, 3] = -current_pose[2, 3]
			# current_pose = homogenous_matrix(current_pose)
			curr_pose_homo = np.vstack((current_pose, [0,0,0,1]))
			prev_pose_homo = np.vstack((prev_pose, [0,0,0,1]))
			prev_pose_homo = np.matmul(prev_pose_homo, curr_pose_homo)

			new_x, new_y, new_z = prev_pose_homo[:3, 3]
			prev_pose = prev_pose_homo[:3, :]

			plt.scatter(new_x, -new_z, color='r')
			plt.savefig("./ransac_changed/" + str(i) + ".png")
