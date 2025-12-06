"""
FIXED main application with Auto-Calibration and Rolling Graph
"""
import cv2
import time
import numpy as np
import json
import os

# Import config and modules
import config
from hand_detector import HandDetector
# Import helpers including new RollingGraph
from utils import draw_text_center, draw_fps, get_state_color, IPWebcamStream, WebcamVideoStream, RollingGraph
from distance_utils import load_calibration, save_calibration, compute_focal_length

class HandTrackingApp:
    """Fixed hand tracking application with smart features."""
    
    def __init__(self):
        """Initialize."""
        self.detector = HandDetector(config)
        
        print(f"\n{'='*60}")
        print("HAND TRACKING - PROTOTYPE v2.0")
        print(f"{'='*60}\n")
        
        # Initialize camera
        print("Initializing camera...")
        if config.USE_IP_WEBCAM:
            print(f"Using IP Webcam: {config.IP_WEBCAM_URL}")
            if config.USE_THREADED_CAPTURE:
                self.camera = IPWebcamStream(config.IP_WEBCAM_URL).start()
                time.sleep(2.0)
            else:
                self.camera = cv2.VideoCapture(config.IP_WEBCAM_URL)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            print(f"Using webcam: {config.REGULAR_WEBCAM_INDEX}")
            if config.USE_THREADED_CAPTURE:
                self.camera = WebcamVideoStream(src=config.REGULAR_WEBCAM_INDEX, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT).start()
                time.sleep(1.0)
            else:
                self.camera = cv2.VideoCapture(config.REGULAR_WEBCAM_INDEX)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        # Test frame
        ret, test_frame = self.read_frame()
        if not ret or test_frame is None:
            raise RuntimeError("Failed to read from camera!")
        
        print(f"✓ Camera OK: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # Virtual object
        frame_height, frame_width = test_frame.shape[:2]
        self.object_shape = {
            'type': 'circle',
            'center': (frame_width // 2, frame_height // 2),
            'radius': config.VIRTUAL_OBJECT_RADIUS
        }
        
        # Initialize Rolling Graph
        self.graph = RollingGraph(
            width=frame_width, 
            height=config.GRAPH_HEIGHT, 
            max_val=config.GRAPH_MAX_VAL,
            color=config.GRAPH_COLOR,
            bg_color=config.GRAPH_BG_COLOR
        )
        
        # State variables
        self.current_state = 'NO_HAND'
        self.smoothed_distance = float('inf')
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.blink_counter = 0
        self.show_danger = True
        self.trackbars_created = False
        
        # Auto-Calibration State
        self.is_calibrating = False
        self.calib_start_time = 0
        self.calib_box = (
            (frame_width - config.CALIB_BOX_SIZE) // 2,
            (frame_height - config.CALIB_BOX_SIZE) // 2,
            config.CALIB_BOX_SIZE,
            config.CALIB_BOX_SIZE
        )
        
        print("\nControls:")
        print("  q = Quit")
        print("  a = Auto-Calibrate Skin Color (Hold hand in box)")
        print("  t = Toggle manual HSV trackbars")
        print("  d = Toggle debug info")
    
    def read_frame(self):
        return self.camera.read()
    
    def compute_state(self, distance_px):
        if distance_px == float('inf'): return 'NO_HAND'
        abs_dist = abs(distance_px)
        if abs_dist <= config.DANGER_PX: return 'DANGER'
        elif abs_dist <= config.WARNING_PX: return 'WARNING'
        return 'SAFE'
    
    def draw_virtual_object(self, frame, state):
        color = get_state_color(state)
        center = self.object_shape['center']
        radius = self.object_shape['radius']
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.circle(frame, center, radius, color, 6)
        cv2.circle(frame, center, 10, color, -1)
    
    def draw_hand_features(self, frame, features):
        if features['contour'] is None: return
        cv2.drawContours(frame, [features['contour']], -1, (0, 165, 255), 3)
        if features['hull'] is not None: cv2.drawContours(frame, [features['hull']], -1, (0, 255, 100), 2)
        if features['centroid']: cv2.circle(frame, features['centroid'], 10, (0, 255, 255), -1)
        if features['closest_point']:
            cv2.circle(frame, features['closest_point'], 15, (0, 0, 255), -1)
            cv2.line(frame, features['closest_point'], self.object_shape['center'], (0, 255, 255), 5)
        for tip in features['fingertips']:
            cv2.circle(frame, tip, 12, (255, 0, 255), -1)
    
    def draw_ui_overlays(self, frame, state, distance_px):
        state_color = get_state_color(state)
        cv2.rectangle(frame, (5, 5), (320, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (320, 100), state_color, 5)
        cv2.putText(frame, f"STATE: {state}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, state_color, 4)
        if distance_px != float('inf'):
            cv2.putText(frame, f"Dist: {abs(distance_px):.0f}px", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        draw_fps(frame, self.fps)
        
        if state == 'DANGER':
            self.blink_counter += 1
            threshold = max(1, int(self.fps / config.BLINK_FPS))
            if self.blink_counter >= threshold:
                self.show_danger = not self.show_danger
                self.blink_counter = 0
            if self.show_danger: draw_text_center(frame, "DANGER DANGER", 2.0, 5, (0, 0, 255))

    def run_auto_calibration(self, frame):
        """Handle the calibration countdown and logic."""
        elapsed = time.time() - self.calib_start_time
        remaining = config.CALIB_DURATION - elapsed
        
        x, y, w, h = self.calib_box
        
        if remaining > 0:
            # Draw box and countdown
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
            draw_text_center(frame, f"Hold Hand in Box: {remaining:.1f}s", 1.0, 2, (0, 255, 255))
        else:
            # Time up: Calibrate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), -1) # Flash green
            result_str = self.detector.auto_calibrate(frame, self.calib_box)
            print(f"✓ Calibration Complete: {result_str}")
            self.is_calibrating = False

    def run(self):
        try:
            while True:
                ret, frame = self.read_frame()
                if not ret or frame is None: break
                
                # Check for Auto-Calibration Mode
                if self.is_calibrating:
                    # Draw instructions and handle timer
                    self.run_auto_calibration(frame)
                    
                    # Show raw frame during calibration (no tracking)
                    cv2.imshow("Camera", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    continue

                # Normal Tracking Mode
                features = self.detector.get_features(frame, self.object_shape)
                
                # Smoothing
                raw_dist = features['distance_px']
                if raw_dist != float('inf'):
                    alpha = config.DISTANCE_SMOOTHING
                    self.smoothed_distance = raw_dist if self.smoothed_distance == float('inf') else (alpha * self.smoothed_distance) + ((1 - alpha) * raw_dist)
                else: self.smoothed_distance = float('inf')
                
                # Update Graph
                if config.SHOW_GRAPH:
                    self.graph.update(abs(self.smoothed_distance) if self.smoothed_distance != float('inf') else float('inf'))
                    self.graph.draw(frame)

                # State & Draw
                self.current_state = self.compute_state(self.smoothed_distance)
                self.draw_virtual_object(frame, self.current_state)
                self.draw_hand_features(frame, features)
                self.draw_ui_overlays(frame, self.current_state, self.smoothed_distance)
                
                if config.SHOW_DEBUG_INFO:
                    cv2.putText(frame, f"Low: {self.detector.hsv_lower}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

                cv2.imshow("Camera", frame)
                
                # Mask Window
                if self.trackbars_created or config.SHOW_DEBUG_INFO:
                    if features['mask'] is not None:
                         mask_display = cv2.cvtColor(features['mask'], cv2.COLOR_GRAY2BGR)
                         if config.ENABLE_FACE_DETECTION and config.REMOVE_FACE_REGION:
                             cv2.line(mask_display, (0, config.FACE_REGION_HEIGHT), (mask_display.shape[1], config.FACE_REGION_HEIGHT), (0,0,255), 2)
                         cv2.imshow("Mask", mask_display)

                # Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('a'):
                    self.is_calibrating = True
                    self.calib_start_time = time.time()
                elif key == ord('t'):
                    if not self.trackbars_created:
                        cv2.namedWindow('Mask')
                        self.detector.create_trackbars('Mask')
                        self.trackbars_created = True
                elif key == ord('d'): config.SHOW_DEBUG_INFO = not config.SHOW_DEBUG_INFO

                # FPS
                self.frame_count += 1
                if time.time() - self.start_time > 1.0:
                    self.fps = self.frame_count / (time.time() - self.start_time)
                    self.frame_count = 0
                    self.start_time = time.time()
                    
        except KeyboardInterrupt: print("Stopped by user")
        finally: self.cleanup()

    def cleanup(self):
        if hasattr(self, 'camera'): self.camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandTrackingApp()
    app.run()