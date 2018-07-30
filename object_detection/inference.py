import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import glob
import time
import argparse
from multiprocessing import Process, Queue

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_details(args):

	PATH_TO_CKPT = args.frozen_graph
	PATH_TO_LABELS = args.label_map
	NUM_CLASSES = args.num_output_classes
	PATH_TO_TEST_IMAGES_DIR = args.input_dir
	PATH_TO_RESULT_IMAGES_DIR = args.output_dir

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg')))
	JPG_PATHS = [ os.path.basename(path) for path in TEST_IMAGE_PATHS ]
	RESULT_IMAGE_PATHS = [ os.path.join(PATH_TO_RESULT_IMAGES_DIR, jpg_path) for jpg_path in JPG_PATHS ]

	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	return TEST_IMAGE_PATHS, RESULT_IMAGE_PATHS, category_index

def load_model(path):
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(path, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')

	return detection_graph

def feed(queue):

	TEST_IMAGE_PATHS, RESULT_IMAGE_PATHS, _ = load_details(args)
	key = 0
	for image_path, result_path in zip(TEST_IMAGE_PATHS, RESULT_IMAGE_PATHS):
		key+=1
		image_np = cv2.imread(image_path, 1)
		image_np_expanded = np.expand_dims(image_np, axis=0)
		queue.put((image_np, image_np_expanded, result_path, key))


def infer(args, feed_queue, stitch_queue):

	detection_graph = load_model(args.frozen_graph)

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Tensors
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			while True:
				if not feed_queue.empty():
					begin = time.time()
					image_np, image_np_expanded, result_path, key = feed_queue.get()

					(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})

					FPS2 = 1/(time.time() - begin)
					stitch_queue.put((boxes, scores, classes, num, image_np, result_path, FPS2, key))

					
def stitch(queue):

	# Inorder frame stitching.
	TEST_IMAGE_PATHS, RESULT_IMAGE_PATHS, category_index = load_details(args)
	SQ = lambda x: np.squeeze(x)
	total_frames = len(RESULT_IMAGE_PATHS)
	frame50 = time.time()
	process_buffer = {}
	current_frame = 1

	while True:
		if not queue.empty():
			boxes, scores, classes, count, image_np, result_path, FPS2, key = queue.get()
			process_buffer[key] = (boxes, scores, classes, count, image_np, result_path, FPS2)

		# Keeps polling for the next frame
		current_objects = process_buffer.pop(current_frame, None)

		if current_objects is not None:

			begin = time.time()
			(boxes, scores, classes, count, image_np, result_path, FPS2) = current_objects
			boxes, classes, scores = SQ(boxes), SQ(classes).astype(np.int32), SQ(scores)
			vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			boxes,
			classes,
			scores,
			category_index,
			use_normalized_coordinates=True,
			line_thickness=8)

			cv2.imwrite(result_path, image_np)
			if key == 50:
				print("50 Frames -> ", time.time() - frame50)

			FPS = 1 / (time.time() - begin)
			log = 'Images Processed: %d Count: %d FPS: %.2f FPS2: %.2f ' % (key, count, FPS, FPS2)
			print(log,  end = '\n')
			current_frame += 1

			if current_frame == total_frames:
				break

		else:
			continue


if __name__ == "__main__":

	# Uncomment to enable GPUs
	os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

	parser = argparse.ArgumentParser()

	parser.add_argument("--input_dir", help = "Path of the input images directory")
	parser.add_argument("--frozen_graph", help = "Path of the frozen graph model")
	parser.add_argument("--label_map", help = "Path of the label map file")
	parser.add_argument("--output_dir", help = "Path of the output directory")
	parser.add_argument("--num_output_classes", help="Defines the number of output classes",
	                    type=int)

	args = parser.parse_args()

	stitch_queue = Queue()
	feed_queue = Queue()

	# For testing purposes both gpu1, gpu2 are just two cpu processes.
	gpu1 = Process(target=infer, args=(args, feed_queue, stitch_queue))
	gpu2 = Process(target=infer, args=(args, feed_queue, stitch_queue))
	stitch_cpu = Process(target=stitch, args=(stitch_queue,))
	feed_cpu = Process(target=feed, args=(feed_queue,))

	gpu1.start()
	gpu2.start()
	stitch_cpu.start()
	feed_cpu.start()

	gpu1.join()
	gpu2.join()
	stitch_cpu.join()
	feed_cpu.start()
