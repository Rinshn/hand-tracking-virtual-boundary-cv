# ============================================================================
# FIXED config.py - WITH FACE REMOVAL AND BETTER SETTINGS
# ============================================================================
"""
FIXED Configuration - Now removes face and tracks hand only!
"""

# IP Webcam settings
USE_IP_WEBCAM = True
IP_WEBCAM_URL = "http://10.53.184.72:8080/video"
REGULAR_WEBCAM_INDEX = 0

# Camera settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# IMPROVED HSV ranges (works better)
HSV_LOWER = [0, 30, 40]
HSV_UPPER = [30, 255, 255]

# YCrCb - DISABLED by default (too strict)
YCRCB_LOWER = [0, 133, 77]
YCRCB_UPPER = [255, 173, 127]
USE_YCRCB = False  # Keep disabled

# Morphology
MORPH_KERNEL_SIZE = 7
MORPH_ITERATIONS = 2

# Contour filtering
MIN_CONTOUR_AREA = 2000

# EASIER distance thresholds (makes DANGER easier to trigger)
SAFE_PX = 200      # Increased from 160
WARNING_PX = 100   # Increased from 80  
DANGER_PX = 70     # Increased from 40 - MUCH EASIER NOW

# Visual
BLINK_FPS = 4

# Performance
USE_BACKGROUND_SUBTRACTOR = False
USE_THREADED_CAPTURE = True

# FACE REMOVAL - CRITICAL FIX
ENABLE_FACE_DETECTION = True  # NEW: Detect and remove face
REMOVE_FACE_REGION = False    # OLD method disabled
FACE_REGION_HEIGHT = 100

# HAND REGION OF INTEREST - CRITICAL FIX
USE_HAND_ROI = True  # NEW: Only look for hand in specific region
HAND_ROI_LEFT = 0.3   # Hand should be in right 70% of frame
HAND_ROI_RIGHT = 1.0
HAND_ROI_TOP = 0.0
HAND_ROI_BOTTOM = 1.0

# Virtual object
VIRTUAL_OBJECT_RADIUS = 100  # Larger circle

# Smoothing
DISTANCE_SMOOTHING = 0.6  # Less smoothing for faster response

# Gaussian blur
GAUSSIAN_BLUR_KERNEL = 7
