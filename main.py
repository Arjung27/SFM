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
		if m1.distance < 0.75 * m2.distance:
			im1_idx = m1.queryIdx
			im2_idx = m1.trainIdx
			[x1,y1] = kp1[im1_idx].pt
			[x2,y2] = kp2[im2_idx].pt
			img1_points.append([int(x1), int(y1)])
			img2_points.append([int(x2), int(y1)])
			good.append([m1])

	return np.asarray(img1_points), np.asarray(img2_points), good

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
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    # return img1,img2

def fundamental_matrix_ransac(img1_points, img2_points):

	max_iter = 1000
	threshold = 60
	pts1 = img1_points.copy()
	pts2 = img2_points.copy()
	ones = np.ones((img1_points.shape[0], 1))
	print(img1_points.shape, pts1.shape)
	pts1 = np.hstack((pts1, ones))
	pts2 = np.hstack((pts2, ones))
	all_index = np.arange(0, img1_points.shape[0])
	min_error = 10000000
	best_F = []

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

	img1 = cv2.imread('./data/1399381497885112.png', 0)
	img2 = cv2.imread('./data/1399381498322587.png', 0)

	img1_points, img2_points, good = find_features(img1, img2)
	F = fundamental_matrix_ransac(img1_points, img2_points)

	lines1 = cv2.computeCorrespondEpilines(img2_points.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	drawlines(img1, img2, lines1, img1_points, img2_points)
