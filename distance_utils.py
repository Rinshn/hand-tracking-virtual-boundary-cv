import json
import os

# calc focal len
def get_focal(known_w, pix_w, dist):
    return (pix_w * dist) / known_w

# get dist
def get_dist_mm(known_w, pix_w, focal):
    if pix_w == 0: 
        return float('inf')
    return (known_w * focal) / pix_w

# save to json
def save_json(focal, known_w, fname='calib.json'):
    data = {
        'focal': float(focal),
        'w': float(known_w)
    }
    
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("saved calib to " + fname)

# load json
def load_json(fname='calib.json'):
    if not os.path.exists(fname):
        return None
    
    try:
        with open(fname, 'r') as f:
            return json.load(f)
    except:
        print("error loading json")
        return None