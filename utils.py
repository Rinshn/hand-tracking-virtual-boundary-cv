# ============================================================================
# FILE 2: utils.py - SAME AS BEFORE WITH IP WEBCAM SUPPORT
# ============================================================================
"""
Utility functions for drawing and helper operations.
"""
import cv2
import numpy as np


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
    """
    Stream handler for IP Webcam (phone camera).
    Works with IP Webcam app URLs.
    """
    def __init__(self, url):
        """Initialize IP webcam stream."""
        self.url = url
        self.stream = cv2.VideoCapture(url)
        
        # Set buffer size to 1 to reduce latency
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        print(f"IP Webcam stream initialized: {url}")
        
    def start(self):
        """Start the thread to read frames."""
        from threading import Thread
        Thread(target=self.update, daemon=True, args=()).start()
        return self
        
    def update(self):
        """Continuously read frames in background thread."""
        while not self.stopped:
            if not self.ret:
                print("Failed to read from IP webcam, retrying...")
                self.stream.release()
                self.stream = cv2.VideoCapture(self.url)
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.ret, self.frame = self.stream.read()
                
    def read(self):
        """Return the most recent frame."""
        return self.ret, self.frame
        
    def stop(self):
        """Stop the thread and release the camera."""
        self.stopped = True
        self.stream.release()


class WebcamVideoStream:
    """Threaded video stream for regular webcam."""
    def __init__(self, src=0, width=640, height=480):
        """Initialize webcam stream."""
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        """Start the thread to read frames."""
        from threading import Thread
        Thread(target=self.update, daemon=True, args=()).start()
        return self
        
    def update(self):
        """Continuously read frames in background thread."""
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.stream.read()
                
    def read(self):
        """Return the most recent frame."""
        return self.ret, self.frame
        
    def stop(self):
        """Stop the thread and release the camera."""
        self.stopped = True
        self.stream.release()