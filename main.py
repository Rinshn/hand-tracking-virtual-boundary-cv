import cv2
import time
import numpy as np
import config as c
from hand_detector import Detector
import utils as u
import distance_utils as du

class App:
    def __init__(self):
        self.d = Detector()
        
        print("starting...")
        
        if c.use_ip:
            print("trying ip cam: " + c.url)
            if c.threaded:
                self.cap = u.IPStream(c.url).start()
                time.sleep(2)
            else:
                self.cap = cv2.VideoCapture(c.url)
        else:
            print("using webcam")
            if c.threaded:
                self.cap = u.WebCam(c.cam_id, c.width, c.height).start()
                time.sleep(1)
            else:
                self.cap = cv2.VideoCapture(c.cam_id)
        
        ret, frame = self.get_frame()
        if not ret:
            print("cam error")
            return
            
        print(f"cam size: {frame.shape[1]}x{frame.shape[0]}")
        
        h, w = frame.shape[:2]
        self.obj = {
            'type': 'circle',
            'center': (w // 2, h // 2),
            'r': c.radius
        }
        
        # graph stuff
        self.g = u.Graph(w, c.g_height, c.g_max, c.g_color, c.g_bg)
        
        self.st = 'NO_HAND'
        self.smooth_dist = float('inf')
        self.fps_val = 0
        self.frames = 0
        self.t0 = time.time()
        self.blink = 0
        self.danger_show = True
        self.bars_on = False
        
        # calib stuff
        self.calibrating = False
        self.calib_t = 0
        self.box = (
            (w - c.box_size) // 2,
            (h - c.box_size) // 2,
            c.box_size,
            c.box_size
        )
        
        print("controls: q=quit, a=calibrate, t=trackbars, d=debug")
    
    def get_frame(self):
        if c.threaded:
            return self.cap.get()
        return self.cap.read()
    
    def get_state(self, d):
        if d == float('inf'): return 'NO_HAND'
        d = abs(d)
        if d <= c.danger: return 'DANGER'
        if d <= c.warn: return 'WARNING'
        return 'SAFE'
    
    def draw_obj(self, img, st):
        col = u.get_col(st)
        cx, cy = self.obj['center']
        r = self.obj['r']
        
        # draw circle
        over = img.copy()
        cv2.circle(over, (cx, cy), r, col, -1)
        cv2.addWeighted(over, 0.25, img, 0.75, 0, img)
        cv2.circle(img, (cx, cy), r, col, 6)
        cv2.circle(img, (cx, cy), 10, col, -1)
    
    def draw_hand(self, img, data):
        if data['cnt'] is None: return
        cv2.drawContours(img, [data['cnt']], -1, (0, 165, 255), 3)
        if data['hull'] is not None: cv2.drawContours(img, [data['hull']], -1, (0, 255, 100), 2)
        if data['center']: cv2.circle(img, data['center'], 10, (0, 255, 255), -1)
        
        if data['pt']:
            cv2.circle(img, data['pt'], 15, (0, 0, 255), -1)
            cv2.line(img, data['pt'], self.obj['center'], (0, 255, 255), 5)
            
        for tip in data['tips']:
            cv2.circle(img, tip, 12, (255, 0, 255), -1)
    
    def draw_gui(self, img, st, d):
        col = u.get_col(st)
        cv2.rectangle(img, (5, 5), (320, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (5, 5), (320, 100), col, 5)
        cv2.putText(img, f"STATE: {st}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 4)
        
        if d != float('inf'):
            cv2.putText(img, f"Dist: {abs(d):.0f}px", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        u.fps(img, self.fps_val)
        
        if st == 'DANGER':
            self.blink += 1
            if self.blink >= max(1, int(self.fps_val / c.blink_rate)):
                self.danger_show = not self.danger_show
                self.blink = 0
            if self.danger_show: u.text(img, "DANGER DANGER", 2.0, 5, (0, 0, 255))

    def do_calib(self, img):
        dt = time.time() - self.calib_t
        rem = c.calib_time - dt
        x, y, w, h = self.box
        
        if rem > 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 3)
            u.text(img, f"Hold Hand: {rem:.1f}s", 1.0, 2, (0, 255, 255))
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), -1)
            res = self.d.calibrate(img, self.box)
            print(f"done: {res}")
            self.calibrating = False

    def run(self):
        try:
            while True:
                ret, img = self.get_frame()
                if not ret or img is None: break
                
                if self.calibrating:
                    self.do_calib(img)
                    cv2.imshow("Cam", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    continue

                # process
                data = self.d.get_data(img, self.obj)
                
                # smooth
                raw = data['dist']
                if raw != float('inf'):
                    a = c.smooth
                    if self.smooth_dist == float('inf'):
                        self.smooth_dist = raw
                    else:
                        self.smooth_dist = (a * self.smooth_dist) + ((1 - a) * raw)
                else:
                    self.smooth_dist = float('inf')
                
                # graph
                if c.show_graph:
                    val = abs(self.smooth_dist) if self.smooth_dist != float('inf') else float('inf')
                    self.g.add(val)
                    self.g.draw(img)

                # drawing
                self.st = self.get_state(self.smooth_dist)
                self.draw_obj(img, self.st)
                self.draw_hand(img, data)
                self.draw_gui(img, self.st, self.smooth_dist)
                
                if c.debug:
                    cv2.putText(img, str(self.d.lower), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

                cv2.imshow("Cam", img)
                
                if self.bars_on or c.debug:
                    if data['mask'] is not None:
                         m_col = cv2.cvtColor(data['mask'], cv2.COLOR_GRAY2BGR)
                         cv2.imshow("Mask", m_col)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): break
                if k == ord('a'):
                    self.calibrating = True
                    self.calib_t = time.time()
                if k == ord('t'):
                    if not self.bars_on:
                        cv2.namedWindow('Mask')
                        self.d.sliders('Mask')
                        self.bars_on = True
                if k == ord('d'): c.debug = not c.debug

                # fps
                self.frames += 1
                if time.time() - self.t0 > 1.0:
                    self.fps_val = self.frames / (time.time() - self.t0)
                    self.frames = 0
                    self.t0 = time.time()
                    
        except KeyboardInterrupt:
            print("stopped")
        finally:
            self.cap.close() if hasattr(self.cap, 'close') else self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = App()
    app.run()