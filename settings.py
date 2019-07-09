# This is contains the settings used in the rest of the program. These function as constants collected here for
# easier, more convenient access.


# ##### GENERAL #####

# resize the input to this resolution, None to disable
resize_input = (640, 480)

# if we try to detect
detect_markers = True
detect_fingertips = True


# ##### ACTIVE AREA SELECTION #####

# the lowest and highest note we use (use only white keys for this). The pattern is '<note (sing, big letter)><octave>'
note_lowest = 'C3'
note_highest = 'C4'

# how close you have to be to a point to move it instead of placing a new one
point_move_min_distance = 25

# the length of a black key relative to the white keys (1 is the same length, .5 half the length, etc)
black_key_length = .6

# the color of the text and lines. values are (blue, green, red) from 0 to 255
text_color = (255, 225, 0)
line_color = (255, 128, 0)


# ##### IMAGE FILTERING #####

# the min and max hue we allow. this filters out red (as red is at the borders of the HSV color space)
skin_hue_min = 25
skin_hue_max = 230

# the threshold of the first mask (smaller number -> less difference in grayscale required)
mask_threshold = 32

# how big is the kernel we use to reduce the noise from the image? bigger numbers -> less noise, less details
# this should be a positive, odd integer
noise_reduction_outer_kernel_size = 5

# the size of the kernel we use to dilate the mask
# this should be a positive, odd integer
noise_reduction_inner_kernel_size = 5

# the minimum amount of saturation we require (less saturation -> more white/gray/less color)
saturation_threshold = 48

# the radius of the kernel used to remove noise from the saturation mask
# this should be a positive, odd integer
saturation_noise_removal_kernel_radius = 3


# ##### DETECTION: MARKERS #####

# the colors we track in HSV (the H/hue value to be precise) in the interval
# values go from 0 to 255 but don't use those near the edges as they are red
tracked_colors = {0: 41, 1: 153, 2: 212, 3: 135}

# the maximum deviation from the above values we allow
tracked_colors_tolerance = 10


# ##### DETECTION: FINGERTIPS #####

pass


# ##### DEBUGGING #####

# show debug windows?
debug_window_enable = True

# the size of the debug windows
debug_window_size = (640, 480)

# how the windows will be placed (columns, rows)
# even (e.g. the first) rows will be from left to right, the others from right to left
debug_window_placement = (3, 2)

# the height of the title bar of the windows
debug_window_title_height = 25

# automatically set the region of interest if no points are selected
debug_allow_no_points = True


# ##### MISC #####

# just a list of all the notes
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
