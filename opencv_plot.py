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


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--input", default = '/cmlscratch/arjgpt27/projects/Oxford_dataset/stereo/centre/', help = "Path of the images")
	parser.add_argument("--model", default = './model', help = "Path of the images")
	parser.add_argument("--output", default = './plots/', help = "Path to store the images")
	Flags = parser.parse_args()

	prev_pose_cv = np.array([[1, 0, 0, 0],
				  [0, 1, 0, 0],
				  [0, 0, 1, 0]], dtype = np.float32)


	files = np.sort(glob.glob(os.path.join(Flags.input, '*png'), recursive=True))
	cut_files = files[0:400]
	files = np.append(files, cut_files)
	fig = plt.figure()
	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(Flags.model)
	intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

	for i in range(0, len(files) - 1):

		print("Reading Frame ",i)
		img1 = cv2.imread(files[i], 0)
		color_image = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
		undistorted_image1 = UndistortImage(color_image, LUT)
		img1 = cv2.cvtColor(undistorted_image1, cv2.COLOR_BGR2GRAY)

		img2 = cv2.imread(files[i+1], 0)
		color_image = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
		undistorted_image2 = UndistortImage(color_image, LUT)
		img2 = cv2.cvtColor(undistorted_image2, cv2.COLOR_BGR2GRAY)

		img1_feat = img1[150:750, :]
		img2_feat = img2[150:750, :]

		img1_points, img2_points = find_features(img1_feat, img2_feat)

		if (len(img1_points) <= 5) or (len(img2_points) <= 5):
			continue

		###############################################################
		E_cv, mask = cv2.findEssentialMat(img1_points, img2_points, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=0.5)
		_,R,t,_ = cv2.recoverPose(E_cv, img1_points, img2_points, focal=fx, pp=(cx, cy))
		angles = rotationMatrixToEulerAngles(R)
		if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:
		# print(np.linalg.det(R))
			if(np.linalg.det(R) < 0):
				R = -R
				t = -t
				print("Inside")
			current_pose_cv = np.hstack((R, t))
			###############################################################
			# if current_pose_cv[2, 3] > 0:
			# 	current_pose_cv[:, 3] = -current_pose_cv[:, 3]
			# current_pose = homogenous_matrix(current_pose)
			curr_pose_homo_cv = np.vstack((current_pose_cv, [0.0, 0.0, 0.0, 1.0]))
			prev_pose_homo_cv = np.vstack((prev_pose_cv, [0.0, 0.0, 0.0, 1.0]))
			prev_pose_homo_cv = np.matmul(prev_pose_homo_cv, curr_pose_homo_cv)

			new_x_cv, new_y_cv, new_z_cv = prev_pose_homo_cv[:3, 3]
			prev_pose_cv = prev_pose_homo_cv[:3, :]

			plt.scatter(new_x_cv, -new_z_cv, color='r')
			plt.savefig(Flags.output + str(i) + ".png")

	print(new_x_cv, new_y_cv, new_z_cv)
