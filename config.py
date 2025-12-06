# ============================================================================
# FIXED config.py - WITH AUTO-CALIBRATION AND GRAPH SETTINGS
# ============================================================================
"""
FIXED Configuration - Now removes face and tracks hand only!
"""

# IP Webcam settings
USE_IP_WEBCAM = True
IP_WEBCAM_URL = "http://10.45.83.111:8080/video"
REGULAR_WEBCAM_INDEX = 0

# Camera settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# HSV ranges (Default - will be overwritten by Auto-Calibration)
HSV_LOWER = [0, 30, 40]
HSV_UPPER = [30, 255, 255]

# YCrCb - DISABLED by default
YCRCB_LOWER = [0, 133, 77]
YCRCB_UPPER = [255, 173, 127]
USE_YCRCB = False

# Morphology
MORPH_KERNEL_SIZE = 7
MORPH_ITERATIONS = 2

# Contour filtering
MIN_CONTOUR_AREA = 2000

# Distance thresholds
SAFE_PX = 200      # > 200px = SAFE
WARNING_PX = 100   # 100-200px = WARNING
DANGER_PX = 70     # < 70px = DANGER

# Visual
BLINK_FPS = 4

# Performance
USE_BACKGROUND_SUBTRACTOR = False
USE_THREADED_CAPTURE = True

# FACE REMOVAL
ENABLE_FACE_DETECTION = True
REMOVE_FACE_REGION = False
FACE_REGION_HEIGHT = 100

# HAND ROI
USE_HAND_ROI = True
HAND_ROI_LEFT = 0.3
HAND_ROI_RIGHT = 1.0
HAND_ROI_TOP = 0.0
HAND_ROI_BOTTOM = 1.0

# Virtual object
VIRTUAL_OBJECT_RADIUS = 100

# Smoothing
DISTANCE_SMOOTHING = 0.6

# Gaussian blur
GAUSSIAN_BLUR_KERNEL = 7

# Debug
SHOW_DEBUG_INFO = True

# --- NEW FEATURES ---

# Live Graph Settings
SHOW_GRAPH = True
GRAPH_HEIGHT = 100  # Height in pixels
GRAPH_COLOR = (0, 255, 255)  # Yellow
GRAPH_BG_COLOR = (50, 50, 50)  # Dark Gray
GRAPH_MAX_VAL = 400  # Max distance to plot (y-axis scale)

# Auto-Calibration Settings
CALIB_BOX_SIZE = 150  # Size of the box in center of screen
CALIB_DURATION = 3.0  # Seconds to hold hand