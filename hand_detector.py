# ============================================================================
# FIXED hand_detector.py - WITH FACE REMOVAL
# ============================================================================
"""
FIXED HandDetector - Now properly removes face and filters contours!
"""
import cv2
import numpy as np
import config


class HandDetector:
    """Hand detector with face removal and ROI filtering."""
    
    def __init__(self, cfg=None):
        """Initialize with face detection."""
        self.config = cfg if cfg else config
        
        self.hsv_lower = np.array(self.config.HSV_LOWER)
        self.hsv_upper = np.array(self.config.HSV_UPPER)
        self.ycrcb_lower = np.array(self.config.YCRCB_LOWER)
        self.ycrcb_upper = np.array(self.config.YCRCB_UPPER)
        
        kernel_size = self.config.MORPH_KERNEL_SIZE
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        
        # CRITICAL FIX: Initialize face detector
        self.face_cascade = None
        if self.config.ENABLE_FACE_DETECTION:
            try:
                # Try to load Haar Cascade for face detection
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("✓ Face detection enabled - will remove face from mask")
            except:
                print("⚠ Face cascade not found - continuing without face removal")
                self.face_cascade = None
        
        self.bg_subtractor = None
        if self.config.USE_BACKGROUND_SUBTRACTOR:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
        
        self.trackbar_values = {}
        print("HandDetector initialized with improved settings")
    
    def create_trackbars(self, window_name='Mask'):
        """Create trackbars for HSV tuning."""
        cv2.createTrackbar('H_Low', window_name, self.config.HSV_LOWER[0], 179, 
                          lambda x: self.trackbar_values.update({'H_Low': x}))
        cv2.createTrackbar('H_High', window_name, self.config.HSV_UPPER[0], 179,
                          lambda x: self.trackbar_values.update({'H_High': x}))
        cv2.createTrackbar('S_Low', window_name, self.config.HSV_LOWER[1], 255,
                          lambda x: self.trackbar_values.update({'S_Low': x}))
        cv2.createTrackbar('S_High', window_name, self.config.HSV_UPPER[1], 255,
                          lambda x: self.trackbar_values.update({'S_High': x}))
        cv2.createTrackbar('V_Low', window_name, self.config.HSV_LOWER[2], 255,
                          lambda x: self.trackbar_values.update({'V_Low': x}))
        cv2.createTrackbar('V_High', window_name, self.config.HSV_UPPER[2], 255,
                          lambda x: self.trackbar_values.update({'V_High': x}))
        print("HSV Trackbars created")
    
    def update_hsv_from_trackbars(self):
        """Update HSV from trackbars."""
        if self.trackbar_values:
            self.hsv_lower = np.array([
                self.trackbar_values.get('H_Low', self.config.HSV_LOWER[0]),
                self.trackbar_values.get('S_Low', self.config.HSV_LOWER[1]),
                self.trackbar_values.get('V_Low', self.config.HSV_LOWER[2])
            ])
            self.hsv_upper = np.array([
                self.trackbar_values.get('H_High', self.config.HSV_UPPER[0]),
                self.trackbar_values.get('S_High', self.config.HSV_UPPER[1]),
                self.trackbar_values.get('V_High', self.config.HSV_UPPER[2])
            ])
    
    def remove_face_from_mask(self, frame, mask):
        """
        CRITICAL FIX: Detect face and remove it from mask.
        This prevents face from being detected as hand!
        """
        if self.face_cascade is None:
            return mask
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        # Draw black rectangles over detected faces in mask
        for (x, y, w, h) in faces:
            # Expand rectangle slightly to ensure full coverage
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(mask.shape[1], x + w + padding)
            y2 = min(mask.shape[0], y + h + padding)
            
            # Black out face region
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        
        return mask
    
    def apply_hand_roi(self, mask, frame_width, frame_height):
        """
        CRITICAL FIX: Only keep contours in hand region (right side).
        This removes body/torso detection.
        """
        if not self.config.USE_HAND_ROI:
            return mask
        
        # Create ROI mask
        roi_mask = np.zeros_like(mask)
        
        # Define hand region (right side of frame)
        x1 = int(frame_width * self.config.HAND_ROI_LEFT)
        x2 = int(frame_width * self.config.HAND_ROI_RIGHT)
        y1 = int(frame_height * self.config.HAND_ROI_TOP)
        y2 = int(frame_height * self.config.HAND_ROI_BOTTOM)
        
        roi_mask[y1:y2, x1:x2] = 255
        
        # Apply ROI
        return cv2.bitwise_and(mask, roi_mask)
    
    def preprocess(self, frame):
        """IMPROVED preprocessing with face removal."""
        self.update_hsv_from_trackbars()
        
        # Blur
        blur_size = (self.config.GAUSSIAN_BLUR_KERNEL, self.config.GAUSSIAN_BLUR_KERNEL)
        blurred = cv2.GaussianBlur(frame, blur_size, 0)
        
        # HSV mask
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Optional YCrCb
        if self.config.USE_YCRCB:
            ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
            ycrcb_mask = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
            skin_mask = cv2.bitwise_or(hsv_mask, ycrcb_mask)
        else:
            skin_mask = hsv_mask
        
        # Background subtraction
        if self.bg_subtractor:
            fg_mask = self.bg_subtractor.apply(frame)
            skin_mask = cv2.bitwise_and(skin_mask, fg_mask)
        
        # CRITICAL FIX: Remove face from mask
        skin_mask = self.remove_face_from_mask(frame, skin_mask)
        
        # CRITICAL FIX: Apply hand ROI
        skin_mask = self.apply_hand_roi(skin_mask, frame.shape[1], frame.shape[0])
        
        # Morphology
        iterations = self.config.MORPH_ITERATIONS
        mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=iterations)
        mask = cv2.dilate(mask, self.morph_kernel, iterations=1)
        
        return mask
    
    def find_hand_contour(self, mask):
        """Find hand contour with position filtering."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # IMPROVED: Filter by area and position
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.config.MIN_CONTOUR_AREA:
                # Check if contour is in valid region (not too low in frame)
                # This helps reject body/torso
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cy = int(M['m01'] / M['m00'])
                    # Hand should be in upper 80% of frame
                    if cy < mask.shape[0] * 0.8:
                        return contour
        
        return None
    
    def compute_hull_and_defects(self, contour):
        """Compute hull and defects."""
        hull = cv2.convexHull(contour, returnPoints=True)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        
        defects = None
        fingertips = []
        
        if len(hull_indices) > 3 and len(contour) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        
                        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        
                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) if (b * c) != 0 else 0
                        
                        if angle <= np.pi / 2:
                            fingertips.append(start)
            except:
                pass
        
        return {'hull': hull, 'defects': defects, 'fingertips': fingertips}
    
    def closest_point_to_object(self, contour, object_shape):
        """Compute closest point."""
        points = contour.reshape(-1, 2).astype(float)
        
        if object_shape['type'] == 'circle':
            center = np.array(object_shape['center'], dtype=float)
            radius = object_shape['radius']
            
            distances_to_center = np.linalg.norm(points - center, axis=1)
            signed_distances = distances_to_center - radius
            
            min_idx = np.argmin(np.abs(signed_distances))
            closest_point = tuple(points[min_idx].astype(int))
            distance_px = float(signed_distances[min_idx])
            
        elif object_shape['type'] == 'rect':
            x, y, w, h = object_shape['x'], object_shape['y'], object_shape['w'], object_shape['h']
            
            clamped_x = np.clip(points[:, 0], x, x + w)
            clamped_y = np.clip(points[:, 1], y, y + h)
            clamped_points = np.stack([clamped_x, clamped_y], axis=1)
            
            distances = np.linalg.norm(points - clamped_points, axis=1)
            
            min_idx = np.argmin(distances)
            closest_point = tuple(points[min_idx].astype(int))
            distance_px = float(distances[min_idx])
        else:
            raise ValueError(f"Unknown object type: {object_shape['type']}")
        
        x, y, w, h = cv2.boundingRect(contour)
        contour_width_px = w
        
        return {
            'closest_point': closest_point,
            'distance_px': distance_px,
            'contour_width_px': contour_width_px
        }
    
    def get_features(self, frame, object_shape):
        """Extract all features."""
        mask = self.preprocess(frame)
        contour = self.find_hand_contour(mask)
        
        if contour is None:
            return {
                'mask': mask,
                'contour': None,
                'hull': None,
                'centroid': None,
                'closest_point': None,
                'distance_px': float('inf'),
                'contour_width_px': 0,
                'fingertips': []
            }
        
        hull_data = self.compute_hull_and_defects(contour)
        
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = (cx, cy)
        else:
            centroid = None
        
        proximity_data = self.closest_point_to_object(contour, object_shape)
        
        return {
            'mask': mask,
            'contour': contour,
            'hull': hull_data['hull'],
            'centroid': centroid,
            'closest_point': proximity_data['closest_point'],
            'distance_px': proximity_data['distance_px'],
            'contour_width_px': proximity_data['contour_width_px'],
            'fingertips': hull_data['fingertips']
        }