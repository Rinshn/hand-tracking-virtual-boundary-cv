import cv2
import numpy as np
from collections import deque

class Graph:
    def __init__(self, w, h, max_v=400, col=(0, 255, 255), bg=(50, 50, 50)):
        self.w = w
        self.h = h
        self.max_v = max_v
        self.col = col
        self.bg = bg
        self.data = deque([0] * w, maxlen=w)

    def add(self, val):
        if val == float('inf'):
            val = self.max_v
        self.data.append(val)

    def draw(self, img):
        # make empty image
        g_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        g_img[:] = self.bg

        # line in middle
        cv2.line(g_img, (0, self.h // 2), (self.w, self.h // 2), (100, 100, 100), 1)

        pts = []
        for i, v in enumerate(self.data):
            v = min(v, self.max_v)
            y = int(self.h - (v / self.max_v * self.h))
            pts.append((i, y))

        if len(pts) > 1:
            cv2.polylines(g_img, [np.array(pts)], False, self.col, 2)

        cv2.putText(g_img, "Live Dist", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # check sizes
        h, w = img.shape[:2]
        if self.w != w:
            self.w = w
            self.data = deque(list(self.data) + [0]*(w-len(self.data)), maxlen=w)
            return

        # blend
        roi = img[h - self.h:, 0:self.w]
        res = cv2.addWeighted(roi, 0.4, g_img, 0.6, 0)
        img[h - self.h:, 0:self.w] = res

def text(img, txt, scale=2, thick=3, col=(0, 0, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(txt, font, scale, thick)[0]
    x = (img.shape[1] - size[0]) // 2
    y = (img.shape[0] + size[1]) // 2
    cv2.putText(img, txt, (x, y), font, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, txt, (x, y), font, scale, col, thick)

def fps(img, val):
    t = f"FPS: {val:.1f}"
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, t, (img.shape[1] - 150, 30), f, 0.7, (255, 255, 255), 2)
    cv2.putText(img, t, (img.shape[1] - 150, 30), f, 0.7, (0, 0, 0), 1)

def get_col(st):
    if st == 'SAFE': return (0, 255, 0)
    if st == 'WARNING': return (0, 255, 255)
    if st == 'DANGER': return (0, 0, 255)
    return (128, 128, 128)

class IPStream:
    def __init__(self, link):
        self.link = link
        self.cap = cv2.VideoCapture(link)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stop_flag = False
        print("ip cam started")
        
    def start(self):
        from threading import Thread
        Thread(target=self.loop, daemon=True).start()
        return self
        
    def loop(self):
        while not self.stop_flag:
            if not self.ret:
                print("retry ip cam...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.link)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.ret, self.frame = self.cap.read()
                
    def get(self):
        return self.ret, self.frame
        
    def close(self):
        self.stop_flag = True
        self.cap.release()

class WebCam:
    def __init__(self, src=0, w=640, h=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(3, w)
        self.cap.set(4, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stop_flag = False
        
    def start(self):
        from threading import Thread
        Thread(target=self.loop, daemon=True).start()
        return self
        
    def loop(self):
        while not self.stop_flag:
            if not self.ret:
                self.close()
            else:
                self.ret, self.frame = self.cap.read()
                
    def get(self):
        return self.ret, self.frame
        
    def close(self):
        self.stop_flag = True
        self.cap.release()