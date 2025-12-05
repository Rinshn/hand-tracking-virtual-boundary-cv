# ============================================================================
# FILE 3: distance_utils.py - SAME AS BEFORE
# ============================================================================
"""Distance estimation utilities using triangle similarity method."""
import json
import os


def compute_focal_length(known_width_mm, perceived_width_px, distance_mm):
    """Calculate camera focal length for distance estimation."""
    return (perceived_width_px * distance_mm) / known_width_mm


def estimate_distance_mm(known_width_mm, perceived_width_px, focal_length):
    """Estimate distance to object using triangle similarity."""
    if perceived_width_px == 0:
        return float('inf')
    return (known_width_mm * focal_length) / perceived_width_px


def save_calibration(focal_length, known_width_mm, filename='calibration.json'):
    """Save calibration data to JSON file."""
    calibration_data = {
        'focal_length': float(focal_length),
        'known_width_mm': float(known_width_mm)
    }
    
    with open(filename, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"Calibration saved to {filename}")
    print(f"  Focal length: {focal_length:.2f} pixels")
    print(f"  Known width: {known_width_mm} mm")


def load_calibration(filename='calibration.json'):
    """Load calibration data from JSON file."""
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
        print(f"Calibration loaded: F={calibration_data['focal_length']:.2f}px")
        return calibration_data
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None