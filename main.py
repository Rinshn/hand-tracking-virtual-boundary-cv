"""
FIXED main application with proper circle positioning and debug info
"""
import cv2
import time
import numpy as np
import json
import os

# Import config first
import config
from hand_detector import HandDetector


# Utility functions
def draw_text_center(img, text, fontScale=2, thickness=3, color=(0, 0, 255)):
    """Draw centered text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, fontScale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (text_x, text_y), font, fontScale, color, thickness)

def draw_fps(img, fps):
    """Draw FPS."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(img, text, (img.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, text, (img.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

def get_state_color(state):
    """Get state color."""
    colors = {
        'SAFE': (0, 255, 0),
        'WARNING': (0, 255, 255),
        'DANGER': (0, 0, 255),
        'NO_HAND': (128, 128, 128)
    }
    return colors.get(state, (255, 255, 255))


class IPWebcamStream:
    """IP Webcam streamer."""
    def __init__(self, url):
        self.url = url
        self.stream = cv2.VideoCapture(url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        print(f"✓ IP Webcam: {url}")
        
    def start(self):
        from threading import Thread
        Thread(target=self.update, daemon=True, args=()).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.ret:
                print("Reconnecting...")
                self.stream.release()
                self.stream = cv2.VideoCapture(self.url)
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.ret, self.frame = self.stream.read()
                
    def read(self):
        return self.ret, self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()


class WebcamVideoStream:
    """Regular webcam."""
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        from threading import Thread
        Thread(target=self.update, daemon=True, args=()).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.stream.read()
                
    def read(self):
        return self.ret, self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()


class HandTrackingApp:
    """Fixed hand tracking application."""
    
    def __init__(self):
        """Initialize."""
        self.detector = HandDetector(config)
        
        print(f"\n{'='*60}")
        print("HAND TRACKING - FIXED VERSION")
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
                self.camera = WebcamVideoStream(
                    src=config.REGULAR_WEBCAM_INDEX,
                    width=config.FRAME_WIDTH,
                    height=config.FRAME_HEIGHT
                ).start()
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
        
        # CRITICAL FIX: Virtual object positioned in CENTER of visible frame
        frame_height, frame_width = test_frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        self.object_shape = {
            'type': 'circle',
            'center': (center_x, center_y),
            'radius': config.VIRTUAL_OBJECT_RADIUS
        }
        
        print(f"✓ Virtual object at ({center_x}, {center_y}) radius={config.VIRTUAL_OBJECT_RADIUS}")
        print(f"✓ Thresholds: DANGER<{config.DANGER_PX}px, WARNING<{config.WARNING_PX}px, SAFE>{config.SAFE_PX}px")
        
        # State
        self.current_state = 'NO_HAND'
        self.smoothed_distance = float('inf')
        
        # FPS
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Blink
        self.blink_counter = 0
        self.show_danger = True
        
        # Trackbars
        self.trackbars_created = False
        
        print("\nControls:")
        print("  q = Quit")
        print("  t = Toggle trackbars")
        print("  d = Toggle debug mode")
        print("\nMove hand TOWARD center circle to trigger DANGER!\n")
    
    def read_frame(self):
        """Read frame."""
        if config.USE_THREADED_CAPTURE:
            ret, frame = self.camera.read()
        else:
            ret, frame = self.camera.read()
        return ret, frame
    
    def compute_state(self, distance_px):
        """Compute state."""
        if distance_px == float('inf'):
            return 'NO_HAND'
        
        abs_dist = abs(distance_px)
        
        if abs_dist <= config.DANGER_PX:
            return 'DANGER'
        elif abs_dist <= config.WARNING_PX:
            return 'WARNING'
        elif abs_dist <= config.SAFE_PX:
            return 'SAFE'
        else:
            return 'SAFE'
    
    def draw_virtual_object(self, frame, state):
        """Draw virtual object with STRONG visibility."""
        color = get_state_color(state)
        center = self.object_shape['center']
        radius = self.object_shape['radius']
        
        # CRITICAL: Make circle VERY visible with multiple layers
        # Semi-transparent fill
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        # Multiple borders for visibility
        cv2.circle(frame, center, radius, color, 6)
        cv2.circle(frame, center, radius - 3, (255, 255, 255), 2)
        cv2.circle(frame, center, radius + 3, (0, 0, 0), 2)
        
        # Center point
        cv2.circle(frame, center, 10, color, -1)
        cv2.circle(frame, center, 12, (255, 255, 255), 2)
        
        # Debug: Draw center crosshair
        if config.SHOW_DEBUG_INFO:
            cv2.line(frame, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (255, 0, 255), 2)
            cv2.line(frame, (center[0], center[1] - 20), (center[0], center[1] + 20), (255, 0, 255), 2)
    
    def draw_hand_features(self, frame, features):
        """Draw hand features."""
        if features['contour'] is None:
            return
        
        # Contour
        cv2.drawContours(frame, [features['contour']], -1, (0, 165, 255), 3)
        
        # Hull
        if features['hull'] is not None:
            cv2.drawContours(frame, [features['hull']], -1, (0, 255, 100), 2)
        
        # Centroid
        if features['centroid']:
            cv2.circle(frame, features['centroid'], 10, (0, 255, 255), -1)
            cv2.circle(frame, features['centroid'], 12, (0, 0, 0), 2)
        
        # Closest point - MOST IMPORTANT
        if features['closest_point']:
            cv2.circle(frame, features['closest_point'], 15, (0, 0, 255), -1)
            cv2.circle(frame, features['closest_point'], 17, (255, 255, 255), 3)
            
            # Line to object center
            cv2.line(frame, features['closest_point'], 
                    self.object_shape['center'], (0, 255, 255), 5)
        
        # Fingertips - IMPROVED visibility
        for i, fingertip in enumerate(features['fingertips']):
            cv2.circle(frame, fingertip, 12, (255, 0, 255), -1)
            cv2.circle(frame, fingertip, 14, (255, 255, 255), 2)
            if config.SHOW_DEBUG_INFO:
                cv2.putText(frame, str(i+1), fingertip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_ui_overlays(self, frame, state, distance_px, features):
        """Draw UI with debug info."""
        state_color = get_state_color(state)
        
        # State box - LARGER
        cv2.rectangle(frame, (5, 5), (320, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (320, 100), state_color, 5)
        cv2.putText(frame, f"STATE: {state}", (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, state_color, 4)
        
        # Distance - LARGER
        if distance_px != float('inf'):
            dist_text = f"Dist: {abs(distance_px):.0f}px"
            cv2.putText(frame, dist_text, (15, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # FPS
        draw_fps(frame, self.fps)

        # Danger blink logic
        if state == 'DANGER':
            self.blink_counter += 1
            # Toggle every N frames
            threshold = max(1, int(self.fps / config.BLINK_FPS))
            if self.blink_counter >= threshold:
                self.show_danger = not self.show_danger
                self.blink_counter = 0
            
            if self.show_danger:
                draw_text_center(frame, "DANGER DANGER", 2.0, 5, (0, 0, 255))
        
        # Debug info
        if config.SHOW_DEBUG_INFO:
             # Show thresholds at bottom
             cv2.putText(frame, f"DANGER < {config.DANGER_PX}px", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
             cv2.putText(frame, f"SAFE > {config.SAFE_PX}px", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
             if features['contour'] is not None:
                 cv2.putText(frame, f"Area: {cv2.contourArea(features['contour']):.0f}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def run(self):
        """Main application loop."""
        try:
            while True:
                # 1. Read frame
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    print("⚠ Lost connection to camera/stream")
                    break

                # 2. Process hand features
                features = self.detector.get_features(frame, self.object_shape)
                
                # 3. Smoothing distance logic
                raw_dist = features['distance_px']
                if raw_dist != float('inf'):
                    if self.smoothed_distance == float('inf'):
                        self.smoothed_distance = raw_dist
                    else:
                        # Simple exponential moving average
                        alpha = config.DISTANCE_SMOOTHING
                        self.smoothed_distance = (alpha * self.smoothed_distance) + ((1 - alpha) * raw_dist)
                else:
                    self.smoothed_distance = float('inf')

                # 4. Determine State
                self.current_state = self.compute_state(self.smoothed_distance)

                # 5. Draw everything
                # Draw virtual object first (background)
                self.draw_virtual_object(frame, self.current_state)
                
                # Draw hand visuals
                self.draw_hand_features(frame, features)
                
                # Draw UI overlays (text/stats)
                self.draw_ui_overlays(frame, self.current_state, self.smoothed_distance, features)

                # 6. Display Windows
                cv2.imshow("Camera", frame)
                
                # Show mask if requested or if trackbars are active
                if self.trackbars_created or config.SHOW_DEBUG_INFO:
                    if features['mask'] is not None:
                         # Convert to Color for overlay drawing
                         mask_display = cv2.cvtColor(features['mask'], cv2.COLOR_GRAY2BGR)
                         # Draw ROI or Face zones if needed
                         if config.ENABLE_FACE_DETECTION and config.REMOVE_FACE_REGION:
                             cv2.line(mask_display, (0, config.FACE_REGION_HEIGHT), 
                                     (mask_display.shape[1], config.FACE_REGION_HEIGHT), (0,0,255), 2)
                         cv2.imshow("Mask", mask_display)

                # 7. Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    if not self.trackbars_created:
                        cv2.namedWindow('Mask')
                        self.detector.create_trackbars('Mask')
                        self.trackbars_created = True
                        print("✓ Trackbars enabled on Mask window")
                elif key == ord('d'):
                    config.SHOW_DEBUG_INFO = not config.SHOW_DEBUG_INFO
                    print(f"Debug mode: {config.SHOW_DEBUG_INFO}")

                # 8. Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()

        except KeyboardInterrupt:
            print("Stopped by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        print("Cleaning up...")
        if hasattr(self, 'camera'):
            self.camera.stop()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    app = HandTrackingApp()
    app.run()
    