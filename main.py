#!/usr/bin/env python3

# import from std lib
import logging
import time
import sys
import traceback
import math
from typing import Union, Tuple

# import 3rd party libs
import cv2 as cv
import numpy as np

# import own modules
import settings


Point = Tuple[float, float]


class Key:
	def __init__(self, note: str, octave: int):
		self.note = note
		self.octave = octave
		self.name = note + str(octave)
		self.white = True if len(note) == 1 else False
		self.pos_relative: float = None
		self.pos_near: Point = None
		self.pos_far: Point = None


class Main:
	def __init__(self):
		# create the logger
		logging.basicConfig()
		self.log = logging.getLogger('finger_tracker')
		self.log.setLevel(logging.DEBUG)
		
		self.log.debug('using opencv version ' + str(cv.__version__))
		
		# define global variables
		self.cap = None
		self.background = None
		self.region_corners = []  # the border points of our region of interest (ROI)
		self.region_mask = None  # the mask generated from the ROI above
		self.mouse_drag = False  # used for dragging the points of the ROI
		self.keys = {}  # the lines on the center of the keys used for detection
	
		# parse the borders (the min_/max_note are the indices in settings.notes), quick sanity check
		min_note = settings.notes.index(settings.note_lowest[:-1])
		min_octave = int(settings.note_lowest[-1])
		max_note = settings.notes.index(settings.note_highest[:-1])
		max_octave = int(settings.note_highest[-1])
		assert min_octave <= max_octave
		if min_octave == max_octave:
			assert min_note < max_note
		
		# calculate the keys by raising the minimum and writing it in self.keys
		white_keys = 0
		while min_octave < max_octave or (min_octave == max_octave and min_note <= max_note):
			key = Key(settings.notes[min_note], min_octave)
			if key.white:
				white_keys += 1
			self.keys[key.name] = key
			min_note += 1
			if min_note >= len(settings.notes):
				min_note = 0
				min_octave += 1
		
		# calculate the indices of the keys
		i = 0  # to calculate the relative positions of the keys
		key_dist = 1 / white_keys
		for name, key in self.keys.items():
			if key.white:
				key.pos_relative = i + key_dist / 2
				i += key_dist
			else:
				key.pos_relative = i
		
		self.log.debug('the notes we\'re using: {}'.format(self.keys.keys()))
		
		# set the parameters for the blob detector
		detector_params = cv.SimpleBlobDetector_Params()
		detector_params.minDistBetweenBlobs = 5
		detector_params.minRepeatability = 0
		detector_params.minThreshold = 254
		detector_params.maxThreshold = 256
		detector_params.thresholdStep = 3
		detector_params.filterByArea = False
		detector_params.filterByColor = False
		detector_params.filterByCircularity = False
		detector_params.filterByConvexity = False
		detector_params.filterByInertia = False
		
		# create the blob detector
		self.blob_detector = cv.SimpleBlobDetector_create(detector_params)
		
	def load_file(self, path: str):
		"""
		Load a video with opencv.
		
		:param path: the path to the video
		"""
		
		# test if the file exists and write an easier to understand log entry
		try:
			with open(path):
				pass
		except FileNotFoundError:
			self.log.error('the supplied path ({}) does not point to a file'.format(path))
			return
		
		# unload any previously loaded file
		if self.cap is not None:
			self.cap.release()
		
		self.log.debug('loading "' + path + '"')
		
		# get the video
		self.cap = cv.VideoCapture(path)
	
		self.log.debug(
			'video loaded, the resolution of the file is '
			+ str(int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)))
			+ 'x'
			+ str(int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
		)
		
	def setup(self):
		""" Set up the parameters for the detection. """
		
		assert self.cap is not None
		
		# get the first/next frame and use it as the background
		ret, self.background = self.cap.read()
		if not ret:
			raise Exception('unable to read the setup frame')
		
		# resize the background
		if settings.resize_input is not None:
			self.background = cv.resize(self.background, settings.resize_input)
		
		cv.imshow('setup', self.background)
		cv.setMouseCallback('setup', self.setup_mouse_callback, param=self)
		while self.cap.isOpened():
			# create a copy of the image
			img = self.background.copy()

			# set the text
			texts = {
				0: 'set the lower near corner (low side of ' + settings.note_lowest + ')',
				1: 'set the lower far corner (low side of ' + settings.note_lowest + ')',
				2: 'set the higher near corner (high side of ' + settings.note_highest + ')',
				3: 'set the higher far corner (high side of ' + settings.note_highest + ')'
			}
			text = texts.get(len(self.region_corners), 'press space to continue')
			cv.putText(img, text, (25, 25), cv.FONT_HERSHEY_PLAIN, 1, settings.text_color, 1, cv.LINE_AA)
			
			# draw the points of the mask
			for p in self.region_corners:
				cv.circle(img, (p[0], p[1]), 5, settings.line_color, 2)
			
			# draw the lines
			if len(self.region_corners) >= 2:
				cv.line(img, self.region_corners[0], self.region_corners[1], settings.line_color, 1)
			if len(self.region_corners) >= 3:
				cv.line(img, self.region_corners[0], self.region_corners[2], settings.line_color, 1)
			if len(self.region_corners) >= 4:
				cv.line(img, self.region_corners[1], self.region_corners[3], settings.line_color, 1)
				cv.line(img, self.region_corners[2], self.region_corners[3], settings.line_color, 1)
				
				# draw the overlay displaying the calculated positions of the keys
				for name, key in self.keys.items():
					cv.line(img, key.pos_near, key.pos_far, settings.line_color, 2)
			
			# draw the image
			cv.imshow('setup', img)
			
			key = cv.waitKey(int(1000/60)) & 0xFF
			if key == ord(' '):
				break
			elif key == ord('q'):
				return False
		
		cv.destroyWindow('setup')
		
		# debug shortcut to set the ROI to the whole image
		if settings.debug_allow_no_points and len(self.region_corners) == 0:
			height, width = self.background.shape[:2]
			self.region_corners = [(width, 0), (width, height), (0, 0), (0, height)]
		
		# create a black image with the same width and height as the background
		self.region_mask = np.zeros(self.background.shape[:2], np.uint8)
		
		# create the mask for our ROI
		assert len(self.region_corners) >= 4
		corners = np.array(
			[self.region_corners[0], self.region_corners[1], self.region_corners[3], self.region_corners[2]],
			np.int32
		)
		cv.fillPoly(self.region_mask, [corners], 255)
		
		return True
		
	@staticmethod
	def setup_mouse_callback(event, x, y, flags, param):
		"""
		The callback we use in setup to handle mouse events.
		This has to be a static method but we cheat a bit and pass self (i.e. our Main object) to the function through param.
		
		:param event: the opencv mouse event type
		:param x: the x position of the mouse event
		:param y:the y position of the mouse event
		:param flags:
		:param param: the self variable from our Main object
		"""
		
		if event == cv.EVENT_LBUTTONDOWN:
			param.mouse_drag = True
			param.log.debug('mouse event: left button down')
			return
		elif event == cv.EVENT_MOUSEMOVE and param.mouse_drag:
			pass
		elif event == cv.EVENT_LBUTTONUP:
			param.log.debug('mouse event: left button up')
			param.mouse_drag = False
		else:
			return
		
		# check if the event was close to an existing point, move the existing point if so
		for i in range(len(param.region_corners)):
			dx = param.region_corners[i][0] - x
			dy = param.region_corners[i][1] - y
			if math.sqrt(dx**2 + dy**2) <= settings.point_move_min_distance:
				param.region_corners[i] = (x, y)
				
				# update the positions of the keys if we have enough points
				if len(param.region_corners) >= 4:
					param.keys_update_positions()
				
				# we moved a point, return
				return
		
		# test if we already have enough points
		if len(param.region_corners) >= 4:
			param.log.info('enough points defined, not placing another one')
			return
		
		# place a new point
		param.region_corners.append((x, y))
		
		# update the positions of the keys if we have enough points
		if len(param.region_corners) >= 4:
			param.keys_update_positions()
		
	@staticmethod
	def _interpolate_linear_2d(p1: Point, p2: Point, x) -> Point:
		"""
		Simple helper to do a linear interpolation between two points in 2D.
		
		:param p1: first point
		:param p2: second point
		:param x: position of the interpolated point (0.0: first, 1.0: second)
		:return: position of the interpolated point
		"""
		
		p_0 = Main._interpolate_linear(p1[0], p2[0], x)
		p_1 = Main._interpolate_linear(p1[1], p2[1], x)
		return p_0, p_1
		
	@staticmethod
	def _interpolate_linear(a: float, b: float, x: float) -> float:
		""" Simple helper to do a linear interpolation between two numbers. """
		
		return int(a + (b - a) * x)
	
	def keys_update_positions(self):
		""" Update the positions of the middle lines for the keys """
		
		for name, key in self.keys.items():
			# calculate the points of the line along the middle of the key
			key.pos_near = self._interpolate_linear_2d(self.region_corners[0], self.region_corners[2], key.pos_relative)
			key.pos_far = self._interpolate_linear_2d(self.region_corners[1], self.region_corners[3], key.pos_relative)
			
			# shorten the line for black keys
			if not key.white:
				key.pos_near = self._interpolate_linear_2d(key.pos_far, key.pos_near, settings.black_key_length)
		
	def set_position(self, minutes: float, seconds: float):
		"""
		Set the current time/playback position in the loaded video.
		
		:param minutes:
		:param seconds:
		"""
		
		assert self.cap is not None
		self.log.debug('setting the current playback position to {}:{:02}'.format(minutes, seconds))
		
		# calculate the timestamp in milliseconds
		ms = minutes * 60000 + seconds * 1000
		
		# set the time
		self.cap.set(cv.CAP_PROP_POS_MSEC, ms)
		
	def run(self):
		""" The part where all the fun stuff happens. Analyses the video and tries to correlate it with the supplied log. """
		
		assert self.cap is not None
		
		pause = False
		while self.cap.isOpened():
			# check for keyboard events
			key = cv.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord(' '):
				pause = not pause
				
			# skip the rest if we paused
			if pause:
				time.sleep(.0167)  # this should give us ~60 cycles per second
				continue
			
			# timestamp when we started working with the frame
			last_frame_time = time.time()
			
			# get the next frame, skip it or terminate when we couldn't grab it
			ret, frame = self.cap.read()
			if not ret:
				if self._get_video_relative_pos() > 99:
					self.log.debug('we seem to have reached the end of the file, terminating the loop')
					break
				self.log.warning('unable to get the current frame, skipping it')
			
			if settings.resize_input is not None:
				# TODO: only resize if the original is larger than the target size
				frame = cv.resize(frame, settings.resize_input)
			
			# generate the mask
			mask = self._generate_mask(frame)
			
			# convert to HSV
			frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)
			
			# apply the mask
			frame_hsv_masked = cv.bitwise_and(frame_hsv, frame_hsv, mask=mask)
			
			# split the channels
			hue, saturation, value = cv.split(frame_hsv_masked)
			
			# remove the areas with very low saturation (i.e. more or less white/gray)
			ret, saturation_mask = cv.threshold(saturation, settings.saturation_threshold, 255, cv.THRESH_BINARY)
			hue = cv.bitwise_and(hue, hue, mask=saturation_mask)
			
			# create the dict for the debug windows and add some stuff to it if enabled
			debug_windows = {}
			if settings.debug_window_enable:
				debug_windows['original'] = frame
				debug_windows['mask'] = mask
				debug_windows['saturation_mask'] = saturation_mask
				debug_windows['hue'] = hue
			
			# create variables to see what methods succeeded
			detect_markers_success = False
			detect_fingertips_success = False
			
			# create the dict we eventually fill when detecting the markers
			tracked_colors_masks = {}
			
			# routine to detect the markers
			if settings.detect_markers:
				# create a version of hue without skin colored things
				hue_in_range = cv.inRange(hue, settings.skin_hue_min, settings.skin_hue_max)
				
				# generate the image without skin colored things
				hue_no_skin = cv.bitwise_and(hue, hue_in_range)
				
				# generate masks for certain hues (within a specified tolerance)
				tracked_colors_masks = self._generate_tracked_colors_masks(hue_no_skin)
				
				# get the keypoints for the markers in the image
				color_blobs = self._detect_markers(tracked_colors_masks)
				# TODO: determine if we succeeded in detecting the markers
				
				# convert the keypoints to something we can view and add them to the debug windows (if they are enabled)
				if settings.debug_window_enable:
					markers = self._keypoints_to_image(color_blobs, hue.shape)
					debug_windows['markers'] = markers
			
			# routine to detect fingertips
			if settings.detect_fingertips:
				# get the list of only the masks (we no longer care about their names at this point)
				tracked_colors_masks = list(tracked_colors_masks.values())
				
				# generate and add masks for the skin colors (remember: reds are at the borders of the hue range in HSV)
				tracked_colors_masks.append(cv.inRange(hue, 1, settings.skin_hue_min))
				tracked_colors_masks.append(cv.inRange(hue, settings.skin_hue_max, 255))
				
				# combine all the masks of the relevant colors
				combined_mask = tracked_colors_masks[0]
				for i in range(1, len(tracked_colors_masks)):
					combined_mask = cv.bitwise_or(combined_mask, tracked_colors_masks[i])
					
				# display the combined mask if debug windows are enabled
				if settings.debug_window_enable:
					debug_windows['combined_mask'] = combined_mask
					
				# TODO: remove, debug
				b, g, r = cv.split(frame)
				rb = cv.absdiff(r, b)
				rg = cv.absdiff(r, g)
				combined = cv.add(rb, rg)
				debug_windows['absdiff combined'] = combined
				
				ret, combined = cv.threshold(combined, 64, 255, cv.THRESH_BINARY)
				debug_windows['absdiff combined threshold'] = combined
				
				# TODO: rotate the image according to the ROI (try cv.RotationWarper or cv.warpAffine)
				# details: https://opencvexamples.blogspot.com/2014/01/rotate-image.html
				# TODO: get contours with opencv
				# TODO: interpret the contours as a graph.
				# TODO: Undistort the data points? (this may be hard, we may get overlaps in the graph)
				# TODO: find strongest peaks (elevation relative to surroundings) in the graph
				
				# TODO: set the variable to indicate success
				
			# TODO: evaluate the results from the different methods
			
			# show and position the different debug windows
			i = 0
			for name, img in debug_windows.items():
				# create the window
				cv.namedWindow(name, cv.WINDOW_NORMAL)
				cv.resizeWindow(name, *settings.debug_window_size)
				cv.imshow(name, img)
				
				# move the window
				pos_y = math.floor(i / settings.debug_window_placement[0]) % settings.debug_window_placement[1]
				pos_x = i % settings.debug_window_placement[0]
				if pos_y % 2 == 1:  # invert pos_x if we're not in the first row
					pos_x = settings.debug_window_placement[0] - pos_x - 1
				pos_x *= settings.debug_window_size[0]
				pos_y *= settings.debug_window_size[1] + settings.debug_window_title_height
				cv.moveWindow(name, pos_x, pos_y)
				
				i += 1
			
			# calculate how long much time we needed for this frame
			frame_duration = time.time() - last_frame_time
			frame_freq = 1 / frame_duration
			
			# log the information from this frame
			# TODO: write this info in the title of the main window (original in debug) as well
			self.log.debug('fps: {:.3f} (duration: {:.5f}s), {:.2f}% through the video'.format(
				frame_freq,
				frame_duration,
				self._get_video_relative_pos()
			))
			
		self.cap.release()
		cv.destroyAllWindows()
		
	def _generate_mask(self, frame):
		""" Generates a mask for the given frame by comparing it with the saved background. """
		
		# subtract the background we captured earlier
		mask = cv.absdiff(frame, self.background)
		
		# convert the mask to grayscale and threshold it
		mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
		ret, mask = cv.threshold(mask, settings.mask_threshold, 255, cv.THRESH_BINARY)
		
		# reduce our mask to only include our ROI
		mask = cv.bitwise_and(mask, self.region_mask)
		
		# remove the noise outside of the mask (with opening)
		r = settings.noise_reduction_outer_kernel_size
		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (r, r))
		mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
		
		# remove noise from inside the mask (with closing)
		r = settings.noise_reduction_inner_kernel_size
		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (r, r))
		mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
		
		return mask
	
	@staticmethod
	def _generate_tracked_colors_masks(hue):
		""" Generate masks for the provided image (hue part only) with the tracked colors withing a certain tolerance. """
		
		tracked_colors_masks = {}
		for name, color in settings.tracked_colors.items():
			# calculate the color boundaries
			color_min = color - settings.tracked_colors_tolerance
			color_min = max(color_min, 0)
			color_max = color + settings.tracked_colors_tolerance
			color_max = min(color_max, 255)
			
			# save the regions with colors within the boundaries
			tracked_colors_masks[name] = cv.inRange(hue, color_min, color_max)
			
		return tracked_colors_masks
	
	def _detect_markers(self, masks):
		""" Detects the colors set in settings and returns a dict with a list of keypoints for each color. """
		
		# find the blobs of the different colors
		color_blobs = {}
		for name, mask in masks.items():
			# detect the blobs (we get a KeyPoint for every detected blob)
			color_blobs[name] = self.blob_detector.detect(mask)
			
			# TODO: show only the single largest Keypoint for every color
			
		return color_blobs  # return the masks as well to allow later use in the fingertip detection
	
	@staticmethod
	def _keypoints_to_image(color_blobs, shape):
		# generate an image for each of the colors and combine them to get a nice image
		color_regions_stitched = None
		for name, color in settings.tracked_colors.items():
			keypoints = color_blobs[name]
			img = Main._draw_keypoints(np.zeros(shape, np.uint8), keypoints, color)
			if color_regions_stitched is None:
				color_regions_stitched = img
			else:
				color_regions_stitched = cv.bitwise_or(img, color_regions_stitched)
		
		# generate a mask from out stitched image we can use for S and V in our HSV image and convert it back to BGR
		ret, result_mask = cv.threshold(color_regions_stitched, 1, 255, cv.THRESH_BINARY)
		result = cv.merge([
			color_regions_stitched,
			result_mask,
			result_mask
		])
		return cv.cvtColor(result, cv.COLOR_HSV2BGR_FULL)
		
	@staticmethod
	def _draw_keypoints(image, keypoints, color):
		"""
		The function for this in opencv doesn't seem to work as expected so I created this little helper.
		This draws every Keypoint as a circle and its center.
		
		:param image: the image we draw the points and circles on
		:param keypoints: a list of Keypoints
		:param color: the color we use to draw the Keypoints
		:return:
		"""
		
		img = image.copy()
		for kp in keypoints:
			x = int(kp.pt[0])
			y = int(kp.pt[1])
			cv.circle(img, (x, y), 2, color, -1)
			cv.circle(img, (x, y), int(kp.size), color, 1)
		return img
		
	def _get_video_relative_pos(self):
		"""
		Calculate and return the current position in the video in percent (0: start, 100: end).
		
		:return: the current relative position in the video
		"""
		
		assert self.cap is not None
		return self.cap.get(cv.CAP_PROP_POS_FRAMES) / self.cap.get(cv.CAP_PROP_FRAME_COUNT) * 100


if __name__ == '__main__':
	# sanity check for the required arguments
	if len(sys.argv) < 2:
		sys.stderr.write('You have to pass the path to the video you want to open as the first argument.')
	else:
		# TODO: write something to overwrite the settings from settings.py with arguments from argv
		m = Main()
		# noinspection PyBroadException
		try:
			m.load_file(sys.argv[1])
			if m.setup():
				m.set_position(5, 7)
				m.run()
		except Exception:
			m.log.error(str(sys.exc_info()[0]) + ': ' + str(sys.exc_info()[1]))
			stack_ = ''
			for stack_line_ in traceback.format_tb(sys.exc_info()[2]):
				stack_ += stack_line_
			m.log.error('stacktrace:\n' + stack_)
			
		m.cap.release()
		cv.destroyAllWindows()
