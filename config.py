# config file

# webcam stuff change it to false for regular webcam
use_ip = True
url = "http://10.45.83.111:8080/video"  # e.g., "http://
cam_id = 0

# screen size
width = 640
height = 480

# color range
lower_color = [0, 30, 40]
upper_color = [30, 255, 255]

# extra color stuff (not used)
ycrcb_min = [0, 133, 77]
ycrcb_max = [255, 173, 127]
use_ycrcb = False

# image processing
k_size = 7
iters = 2
min_area = 2000

# distances
safe = 200
warn = 100
danger = 70

# blink speed
blink_rate = 4

# settings
bg_sub = False
threaded = True
face_detect = True
remove_face = False
face_h = 100

# roi
use_roi = True
roi_l = 0.3
roi_r = 1.0
roi_t = 0.0
roi_b = 1.0

# circle size
radius = 100

# smooth factor
smooth = 0.6

# blur
blur_k = 7

# debug
debug = True

# graph
show_graph = True
g_height = 100
g_color = (0, 255, 255)
g_bg = (50, 50, 50)
g_max = 400

# calibration
box_size = 150
calib_time = 3.0