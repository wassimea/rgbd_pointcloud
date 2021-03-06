from utils_camera import *

import os
import math

import cv2
import numpy as np

import pyrealsense2 as rs

def colorize_depthmap(img_depth):
    ## colorize depth map for easy visualization
    img_depth_normalized = cv2.normalize(img_depth.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX) # convert to normalized floating point
    img_depth_grayscale = img_depth_normalized * 255 # now to grayscale
    img_depth_clr = cv2.applyColorMap(img_depth_grayscale.astype(np.uint8), cv2.COLORMAP_JET) # apply the color mapping
    return img_depth_clr

def image_fusion(camera_params, depthData, clrImg=None):
	"""
		Given a depth image and its corresponding color image, return a colored point cloud as a vector of (x, y, z, r, g, b).
		Assume only depth and color.
		The output format is a PLY (required to view it in color in MeshLab).
	"""

	numberOfVertices = depthData.size

	h, w = depthData.shape

	# generate point cloud via numpy array functions
	coords = np.indices((h, w))
	
	# geometry
	xcoords = (((coords[1] - camera_params.cx)/camera_params.fx)*depthData).flatten()
	ycoords = (((coords[0] - camera_params.cy)/camera_params.fy)*depthData).flatten()
	zcoords = depthData.flatten()
	
	# color
	chan_red = chan_blue = chan_green = None

	chan_red = clrImg[..., 2].flatten()
	chan_blue = clrImg[..., 1].flatten()
	chan_green = clrImg[..., 0].flatten()

	ptcloud = None


	ptcloud = np.dstack((xcoords, ycoords, zcoords, chan_red, chan_blue, chan_green))[0]

	return ptcloud, numberOfVertices

def output_pointcloud(nVertices, ptcloud, strOutputPath):
	"""
		Given a point cloud produced from image_fusion, output it to a PLY file.
	"""
	# open the file and write out the standard ply header
	outputFile = open(strOutputPath + ".ply", "w")
	outputFile.write("ply\n")
	outputFile.write("format ascii 1.0\n")
	outputFile.write("comment generated via python script Process3DImage\n")
	outputFile.write("element vertex %d\n" %(nVertices))
	outputFile.write("property float x\n")
	outputFile.write("property float y\n")
	outputFile.write("property float z\n")

	outputFile.write("property uchar red\n")
	outputFile.write("property uchar green\n")
	outputFile.write("property uchar blue\n")
	outputFile.write("element face 0\n")
	outputFile.write("property list uchar int vertex_indices\n")
	outputFile.write("end_header\n")

	# output the actual points
	for pt in ptcloud:
		dx, dy, dz = pt[0:3]

		dx *= 0.001
		dy *= 0.001
		dz *= 0.001

		r, g, b = pt[3:]
		outputFile.write("%10.6f %10.6f %10.6f %d %d %d\n" %(dx, dy, dz, r, g, b))

	outputFile.close()

def get_color_depth_frames():

	pipeline = rs.pipeline()
	
	config = rs.config()
	config.enable_stream(rs.stream.color, width=640, height=480)
	config.enable_stream(rs.stream.depth, width=640, height=480)

	pipeline.start(config)
	for i in range(100):
		frames = pipeline.wait_for_frames()

		color = frames.first(rs.stream.color)
		depth = frames.first(rs.stream.depth)
		color = np.asanyarray(color.get_data())
		color = color[...,::-1]
		depth = np.asanyarray(depth.get_data())
		depth_vis = colorize_depthmap(depth)

		cv2.imshow("color", color)
		cv2.imshow("depth_vis", depth_vis)
		cv2.waitKey(1)
	return color, depth



if __name__ == "__main__":
	out = "/home/wassimea/Desktop/cloud"

	img_color, img_depth = get_color_depth_frames()

	params = GetCameraParameters("RealSenseD435", 1.0)
	ptcloud, nVertices = image_fusion(params, img_depth, img_color)
	output_pointcloud(nVertices, ptcloud, out)


