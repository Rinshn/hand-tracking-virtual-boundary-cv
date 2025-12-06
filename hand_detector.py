import cv2
import numpy as np
import config as c

class Detector:
    def __init__(self):
        self.lower = np.array(c.lower_color)
        self.upper = np.array(c.upper_color)
        
        k = c.k_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        
        self.face = None
        if c.face_detect:
            try:
                # loading face xml
                p = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face = cv2.CascadeClassifier(p)
                print("face detect on")
            except:
                print("no face xml found")
                self.face = None
        
        self.bars = {}
        print("detector ready")
    
    def sliders(self, win='Mask'):
        cv2.createTrackbar('H_min', win, self.lower[0], 179, lambda x: self.bars.update({'H_min': x}))
        cv2.createTrackbar('H_max', win, self.upper[0], 179, lambda x: self.bars.update({'H_max': x}))
        cv2.createTrackbar('S_min', win, self.lower[1], 255, lambda x: self.bars.update({'S_min': x}))
        cv2.createTrackbar('S_max', win, self.upper[1], 255, lambda x: self.bars.update({'S_max': x}))
        cv2.createTrackbar('V_min', win, self.lower[2], 255, lambda x: self.bars.update({'V_min': x}))
        cv2.createTrackbar('V_max', win, self.upper[2], 255, lambda x: self.bars.update({'V_max': x}))
        print("sliders made")
    
    def check_sliders(self):
        if self.bars:
            self.lower = np.array([
                self.bars.get('H_min', self.lower[0]),
                self.bars.get('S_min', self.lower[1]),
                self.bars.get('V_min', self.lower[2])
            ])
            self.upper = np.array([
                self.bars.get('H_max', self.upper[0]),
                self.bars.get('S_max', self.upper[1]),
                self.bars.get('V_max', self.upper[2])
            ])

    def calibrate(self, img, rect):
        x, y, w, h = rect
        roi = img[y:y+h, x:x+w]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        h_med = np.median(hsv[:, :, 0])
        s_med = np.median(hsv[:, :, 1])
        v_med = np.median(hsv[:, :, 2])
        
        print(f"calib: {h_med} {s_med} {v_med}")
        
        # set ranges
        off_h = 20
        
        s_low = max(20, s_med - 50)
        v_low = max(20, v_med - 50)
        
        h_min = max(0, h_med - off_h)
        h_max = min(179, h_med + off_h)
        
        self.lower = np.array([h_min, s_low, v_low], dtype=np.uint8)
        self.upper = np.array([h_max, 255, 255], dtype=np.uint8)
        
        # update slider vals
        if 'H_min' in self.bars:
            self.bars['H_min'] = int(h_min)
            self.bars['H_max'] = int(h_max)
            self.bars['S_min'] = int(s_low)
            self.bars['S_max'] = 255
            self.bars['V_min'] = int(v_low)
            self.bars['V_max'] = 255

        return f"H:{int(h_med)}"

    def no_face(self, img, m):
        if self.face is None: return m
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.face.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
        for (x, y, w, h) in rects:
            pad = 20
            cv2.rectangle(m, (max(0, x-pad), max(0, y-pad)), (min(m.shape[1], x+w+pad), min(m.shape[0], y+h+pad)), 0, -1)
        return m
    
    def cut_roi(self, m, w, h):
        if not c.use_roi: return m
        mask2 = np.zeros_like(m)
        x1 = int(w * c.roi_l)
        x2 = int(w * c.roi_r)
        y1 = int(h * c.roi_t)
        y2 = int(h * c.roi_b)
        mask2[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(m, mask2)
    
    def process(self, img):
        self.check_sliders()
        b = (c.blur_k, c.blur_k)
        blur = cv2.GaussianBlur(img, b, 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        
        mask = self.no_face(img, mask)
        mask = self.cut_roi(mask, img.shape[1], img.shape[0])
        
        it = c.iters
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=it)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=it)
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        return mask
    
    def get_cnt(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for x in cnts:
            if cv2.contourArea(x) >= c.min_area:
                M = cv2.moments(x)
                if M['m00'] != 0:
                    cy = int(M['m01'] / M['m00'])
                    if cy < mask.shape[0] * 0.8:
                        return x
        return None
    
    def get_hull(self, cnt):
        pts = cv2.convexHull(cnt, returnPoints=True)
        idx = cv2.convexHull(cnt, returnPoints=False)
        tips = []
        if len(idx) > 3 and len(cnt) > 3:
            try:
                defs = cv2.convexityDefects(cnt, idx)
                if defs is not None:
                    for i in range(defs.shape[0]):
                        s, e, f, d = defs[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        
                        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        
                        ang = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) if (b * c) != 0 else 0
                        if ang <= np.pi / 2:
                            tips.append(start)
            except: pass
        return {'hull': pts, 'tips': tips}
    
    def get_dist(self, cnt, obj):
        pts = cnt.reshape(-1, 2).astype(float)
        cx, cy = obj['center']
        r = obj['r']
        
        # calc dists
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        res = dists - r
        
        idx = np.argmin(np.abs(res))
        pt = tuple(pts[idx].astype(int))
        val = float(res[idx])
        
        _, _, w, _ = cv2.boundingRect(cnt)
        return {'pt': pt, 'val': val, 'w': w}
    
    def get_data(self, img, obj):
        m = self.process(img)
        cnt = self.get_cnt(m)
        
        if cnt is None:
            return {'mask': m, 'cnt': None, 'hull': None, 'center': None,
                    'pt': None, 'dist': float('inf'), 'w': 0, 'tips': []}
        
        hull = self.get_hull(cnt)
        M = cv2.moments(cnt)
        c_pt = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) if M['m00'] != 0 else None
        prox = self.get_dist(cnt, obj)
        
        return {'mask': m, 'cnt': cnt, 'hull': hull['hull'], 'center': c_pt,
                'pt': prox['pt'], 'dist': prox['val'],
                'w': prox['w'], 'tips': hull['tips']}