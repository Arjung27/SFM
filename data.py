import numpy as np
import cv2
import glob
import os
import argparse
from ReadCameraModel import *
from UndistortImage import *

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--input", default = '/media/arjun/My Passport/Oxford_dataset/stereo/centre/', help = "Path of the images")
	parser.add_argument("--model", default = './model', help = "Path of the images")
	parser.add_argument("--output", default= './data/', help = "Path of output directory")
	Flags = parser.parse_args()

	if not os.path.exists(Flags.output):
		os.makedirs(Flags.output)

	files = np.sort(glob.glob(Flags.input + '*.png', recursive=True))

	for count, img in enumerate(files):

		image = cv2.imread(img, 0)
		color_image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
		fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(Flags.model)
		undistorted_image = UndistortImage(color_image, LUT)

		if count < 100:
			filename = os.path.join(Flags.output, img.split('/')[-1])
			cv2.imwrite(filename, undistorted_image)
