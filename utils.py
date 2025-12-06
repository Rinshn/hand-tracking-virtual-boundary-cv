# ============================================================================
# FILE: utils.py - Helpers, Camera, & Rolling Graph
# ============================================================================
"""
Utility functions for drawing, camera, and data visualization.
"""
import cv2
import numpy as np
from collections import deque

class RollingGraph:
    """Draws a live rolling graph of values at the bottom of the screen."""
    def __init__(self, width, height, max_val=400, color=(0, 255, 255), bg_color=(50, 50, 50)):
        self.width = width
        self.height = height
        self.max_val = max_val
        self.color = color
        self.bg_color = bg_color
        # Store history of values (one per pixel of width)
        self.data = deque([0] * width, maxlen=width)

    def update(self, value):
        """Add a new value to the graph."""
        # Handle 'inf' or invalid values by repeating last known or 0
        if value == float('inf'):
            value = self.max_val  # Draw at top if no hand
        self.data.append(value)

    def draw(self, frame):
        """Draw the graph overlay onto the frame."""
        # Create a separate image for the graph
        graph_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        graph_img[:] = self.bg_color

        # Draw grid lines (optional)
        cv2.line(graph_img, (0, self.height // 2), (self.width, self.height // 2), (100, 100, 100), 1)

        # Plot points
        points = []
        for i, val in enumerate(self.data):
            # Normalize value to height (inverted because y=0 is top)
            # Clip value to max range
            val = min(val, self.max_val)
            normalized_h = int(self.height - (val / self.max_val * self.height))
            points.append((i, normalized_h))

        # Draw lines between points
        if len(points) > 1:
            cv2.polylines(graph_img, [np.array(points)], isClosed=False, color=self.color, thickness=2)

        # Overlay text
        cv2.putText(graph_img, "Live Distance Tracking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend into the bottom of the frame
        # Ensure we don't go out of bounds
        h, w = frame.shape[:2]
        if self.width != w:
            # Resize graph if frame width changed
            self.width = w
            self.data = deque(list(self.data) + [0]*(w-len(self.data)), maxlen=w)
            return # Skip this frame to avoid crash

        roi = frame[h - self.height:, 0:self.width]
        result = cv2.addWeighted(roi, 0.4, graph_img, 0.6, 0)
        frame[h - self.height:, 0:self.width] = result

def draw_text_center(img, text, fontScale=2, thickness=3, color=(0, 0, 255)):
    """Draw text centered on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, fontScale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (text_x, text_y), font, fontScale, color, thickness)

def draw_fps(img, fps):
    """Draw FPS counter in top-right corner."""
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (img.shape[1] - 150, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img, text, (img.shape[1] - 150, 30), font, 0.7, (0, 0, 0), 1)

def get_state_color(state):
    """Get BGR color for each state."""
    colors = {
        'SAFE': (0, 255, 0),
        'WARNING': (0, 255, 255),
        'DANGER': (0, 0, 255),
        'NO_HAND': (128, 128, 128)
    }
    return colors.get(state, (255, 255, 255))

class IPWebcamStream:
    """Stream handler for IP Webcam (phone camera)."""
    def __init__(self, url):
        self.url = url
        self.stream = cv2.VideoCapture(url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        print(f"IP Webcam stream initialized: {url}")
        
    def start(self):
        from threading import Thread
        Thread(target=self.update, daemon=True, args=()).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.ret:
                print("Failed to read from IP webcam, retrying...")
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
    """Threaded video stream for regular webcam."""
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