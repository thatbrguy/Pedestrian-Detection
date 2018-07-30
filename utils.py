import os
import cv2
import glob
import math
import matplotlib.pyplot as plt
import skimage
import re
from skimage import transform
import numpy as np
from PIL import Image

def arrange(parent, child):
	"""
	Gives a set of randomly named png files new names in
	ascending order (Starting from 0)
	"""
	x = os.listdir(parent)
	if not os.path.exists(child):
		os.mkdir(child)
	for i,file in enumerate(x):
		newfile = str(i) + '.jpg'
		src = os.path.join(os.path.abspath(parent), file)
		dst = os.path.join(os.path.abspath(child), newfile)
		os.rename(src, dst)

def select(parent, child, num, start):
	"""
	Selects a subset of images from the main dataset and numbers them
	in increasing order.
	"""
	total = len(os.listdir(parent))
	select = np.random.permutation(total)
	select = select[:num]

	files = np.array(os.listdir(parent))[select]

	if not os.path.exists(child):
		os.mkdir(child)

	for i,file in enumerate(files, start):
		newfile = str(i) + '.png'
		src = os.path.join(os.path.abspath(parent), file)
		dst = os.path.join(os.path.abspath(child), newfile)
		os.rename(src, dst)

def choose():
	"""
	Collects PNG files only if a XML with the same
	name exists in the folder
	"""
	xml = glob.glob('./*.xml')
	if not os.path.exists('Select'):
		os.mkdir('Select')
	for file in xml:
		png = file[:-4] + '.png'
		dstxml = os.path.join(os.path.abspath('Select'), file[2:])
		dstpng = os.path.join(os.path.abspath('Select'), png[2:])
		os.rename(file, dstxml)
		os.rename(png, dstpng)

def png2jpg():
	"""
	Converts all PNG files inside a directory to JPGs
	"""
	x = glob.glob('./*png')
	for i,file in enumerate(x):
		img = Image.open(file)
		img.save('./JPG/' + file[2:-4] + '.jpg')
		print(i, end = '/r')

def movejpg():
	"""
	Collects all JPGs inside a directory and moves them to another
	"""
	os.mkdir('JPG')
	x = glob.glob('./*jpg')
	for file in x:
		os.rename(file, './JPG' + file[1:])

def namelist():
	"""
	Creates the trainval.txt text file, assuming your dataset is numbered 
	in a continuous fashion
	"""
	file = open('trainval.txt', 'w')
	for i in range(283):
		file.write(str(i))
		file.write("\n")
	file.close()

def modifyxml():
	"""
	Used to change old image filename inside all XML files to the new 
	image filenames IF you have changed your image filenames to a continuous
	numbered sequence after you have annotated them.
	"""
	for i in range(283):
		with open(str(i) + ('.xml'), 'r') as file:
			data = file.readlines()
		data[2] = '  <filename>' + str(i) + '.jpg' + '</filename>\n'
		with open(str(i) + ('.xml'), 'w') as file:
			file.writelines( data )

def choose1():
	"""
	Collects PNG files only if a XML with the same
	name exists in the folder AND names them in ascending order

	WARNING: You will have to change the image filename element inside
			 the XML file to the new filename, otherwise your program 
			 won't find the corresponding image. Use this function with
			 caution. You can use modifyxml to rectify your mistakes. 
	"""
	xml = glob.glob('./*.xml')
	if not os.path.exists('Select1'):
		os.mkdir('Select1')
	for i,file in enumerate(xml, 0):
		png = file[:-4] + '.png'
		dstxml = os.path.join(os.path.abspath('Select1'), str(i) + '.xml')
		dstpng = os.path.join(os.path.abspath('Select1'), str(i) + '.png')
		os.rename(file, dstxml)
		os.rename(png, dstpng)

def resize():
	"""
	Used to resize images
	"""
	x = os.listdir('images')
	for file in x:
		img = np.array(Image.open('images/' + file))
		shape = np.shape(img)
		if(shape[1] == 1920):
			newshape = (shape[0]//6, shape[1]//6, 3)
		else:
			newshape = (shape[0]//4, shape[1]//4, 3)
		img = skimage.transform.resize(img, newshape, mode = 'constant')
		im = Image.fromarray((img*255).astype(np.uint8))
		im.save('new/'+file)

def resizeann():
	
	"""
	Used to resize the bounding box dimensions inside the
	annotations IF you resized images AFTER annotating them
	"""

	x = glob.glob('./*xml')
	for xml in x:
		
		with open(xml, 'r') as file:
			data = file.readlines()
		
		flag = False	
		for i in range(100):
			
			if(data[i][4:11] == '<width>'):
				width = int(re.findall(r'\d+', data[i])[0])
				height = int(re.findall(r'\d+', data[i+1])[0])
				
				if(width == 1920):
					width = width // 6
					height = height // 6
					flag = True
				else:
					width = width // 4
					height = height // 4

				data[i] = '    <width>' + str(width) + '</width>\n'
				data[i+1] = '    <height>' + str(height) + '</height>\n'

			if(data[i][6:12] == '<xmin>'):
				xmin = int(re.findall(r'\d+', data[i])[0])
				ymin = int(re.findall(r'\d+', data[i+1])[0])
				xmax = int(re.findall(r'\d+', data[i+2])[0])
				ymax = int(re.findall(r'\d+', data[i+3])[0])				
				
				if(flag):
					xmin = xmin // 6
					xmax = xmax // 6
					ymin = ymin // 6
					ymax = ymax // 6
				else:
					xmin = xmin // 4
					xmax = xmax // 4
					ymin = ymin // 4
					ymax = ymax // 4

				data[i] = '      <xmin>' + str(xmin) + '</xmin>\n'
				data[i+1] = '      <ymin>' + str(ymin) + '</ymin>\n'
				data[i+2] = '      <xmax>' + str(xmax) + '</xmax>\n'
				data[i+3] = '      <ymax>' + str(ymax) + '</ymax>\n'

			if(data[i] == '</annotation>\n'):
				break

		with open('new/'+xml[2:], 'w') as file:
			file.writelines(data)

def crop_resize(parent, child):

	"""
	Crops and Resizes an image
	"""
	x = os.listdir(parent)
	if not os.path.exists(child):
		os.mkdir(child)
	for i,file in enumerate(x,1):
		src = os.path.join(os.path.abspath(parent), file)
		dst = os.path.join(os.path.abspath(child), file)
		img = cv2.imread(src, 1)
		shape = np.shape(img)
		if shape[0] != shape[1]:
			if shape[0] > shape[1]:
				dif = shape[0] - shape[1]
				lower_half = int(math.floor(dif / 2))
				upper_half = int(math.ceil(dif / 2))
				img = img[lower_half : shape[0]-upper_half, :, :]

			elif shape[1] > shape[0]:
				dif = shape[1] - shape[0]
				lower_half = int(math.floor(dif / 2))
				upper_half = int(math.ceil(dif / 2))
				img = img[:, lower_half : shape[1]-upper_half, :]

		img = skimage.transform.resize(img, (256,256,3), mode = 'constant')
		img = (img*255).astype(np.uint8)
		print(i, end = '\r')
		cv2.imwrite(dst, img)

def img2numpy(parent):
	"""
	Converts an image dataset to a numpy file
	"""
	x = os.listdir(parent)
	arr = []
	for i,file in enumerate(x,1):
		src = os.path.join(os.path.abspath(parent), file)
		arr.append(plt.imread(src))
		print(i, end = '\r')
	np.save('A.npy', arr)


def video2im(src, dst):
	"""
	Extracts all frames from a video and saves them as jpgs
	"""
    cap = cv2.VideoCapture(src)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    frame = 0
    while True:
        check, img = cap.read()
        if check:
            cv2.imwrite(os.path.join(dst,"%d.jpg") %frame, img)
            frame += 1
            print(frame, end = '\r')
        else:
            break
    cap.release()

from cv2 import VideoWriter, VideoWriter_fourcc
#Comment this if you don't need video processing capability

def im2video(src, output, fps = 30.0, image_size = (1280,586)):
	"""
	Converts JPGs into a video
	"""
	fourcc = VideoWriter_fourcc(*"XVID")
	vid = VideoWriter(output,fourcc, fps, image_size)
	start = 2231 #Assuming images are sequentially numbered
	end = 7391 + 1
	for i in range(start, end):
		path = os.path.join(os.path.abspath(src), str(i) + '.jpg')
		img = cv2.imread(path, 1)
		print(i, end = '\r')
		vid.write(img)
