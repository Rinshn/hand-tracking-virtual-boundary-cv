# Hand Tracking with Virtual Boundary Detection

Real-time hand tracking system that detects when a hand approaches a virtual boundary 
and triggers visual warnings (SAFE → WARNING → DANGER) without using MediaPipe or OpenPose.

## Features

- **Classical Computer Vision**: Uses HSV/YCrCb color segmentation + contour detection
- **Real-time Performance**: Achieves ≥8 FPS on CPU-only execution
- **Distance-based States**: SAFE, WARNING, DANGER zones with visual feedback
- **Calibration Support**: Optional triangle similarity for cm-based distance
- **Interactive Controls**: Runtime HSV tuning via trackbars
- **Threaded Capture**: Optimized camera I/O for maximum FPS

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone or download the project files
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Run the application:
```bash
python main.py
```

### Controls

- **'q'** - Quit application
- **'c'** - Enter calibration mode (for distance estimation in cm)
- **'t'** - Toggle HSV trackbars for runtime color tuning

## How It Works

### 1. Hand Detection Pipeline

The system uses a multi-stage approach:

1. **Preprocessing**:
   - Gaussian blur to reduce noise
   - Convert to HSV and YCrCb color spaces
   - Apply color thresholds to create skin mask
   - Morphological operations (open/close) to clean noise

2. **Contour Analysis**:
   - Find largest contour (assumed to be hand)
   - Compute convex hull for robust shape
   - Calculate centroid (palm center)
   - Find closest point to virtual object

3. **Distance Calculation**:
   - Vectorized numpy operations for speed
   - Signed distance to virtual boundary
   - Exponential smoothing to reduce jitter

4. **State Machine**:
   - **SAFE**: Distance > 160 pixels (green)
   - **WARNING**: Distance 80-160 pixels (yellow)
   - **DANGER**: Distance < 80 pixels (red + blinking alert)

### 2. Why Convex Hull + Contour?

- **Convex hull** provides the outer envelope of the hand, robust to finger positions
- **Closest contour point** gives accurate proximity to the virtual object
- **Centroid alone** would not detect when fingertips approach the boundary
- This method works regardless of hand orientation or finger configuration

## Calibration (Optional)

For distance estimation in centimeters (using triangle similarity):

1. Press **'c'** during operation
2. Place your hand at a known distance from the camera (e.g., 50cm)
3. The system will show your hand width in pixels
4. Enter the following when prompted:
   - **Hand width**: Measure palm width in mm (typically 85-95mm)
   - **Distance**: Distance from camera in mm (e.g., 500mm)
5. Press **'s'** to save calibration

**Example calibration**:
- Hand width: 85 mm
- Distance: 500 mm (50 cm)
- Perceived width: 120 pixels
- Focal length = (120 × 500) / 85 = 705.88 pixels

Calibration is saved to `calibration.json` and loaded on startup.

## Troubleshooting

### Low FPS (< 8 FPS)

1. **Reduce frame size**: Edit `config.py` and set `FRAME_WIDTH = 320`, `FRAME_HEIGHT = 240`
2. **Disable background subtraction**: Set `USE_BACKGROUND_SUBTRACTOR = False`
3. **Enable threading**: Set `USE_THREADED_CAPTURE = True` (default)
4. **Reduce morphology iterations**: Edit `hand_detector.py` morphology calls

### Poor Hand Detection

1. **Adjust HSV values**: Press 't' for trackbars, tune H/S/V ranges
2. **Check lighting**: Ensure consistent, bright lighting
3. **Remove background**: Reduce clutter behind hand
4. **Calibrate for skin tone**: Different skin tones may need different HSV ranges

### Common HSV Ranges

- **Light skin**: H: 0-20, S: 20-150, V: 70-255
- **Medium skin**: H: 0-25, S: 40-170, V: 50-255
- **Dark skin**: H: 0-30, S: 30-180, V: 30-255

### Face False Positives

The system automatically removes the top portion of the frame to avoid detecting face as hand. 
Adjust `FACE_REGION_HEIGHT` in `config.py` if needed.

## Performance Optimizations

The code includes several optimizations for real-time performance:

1. **Threaded camera capture** - Offloads I/O to separate thread (~2x FPS boost)
2. **Vectorized numpy operations** - No Python loops for distance calculations
3. **Exponential smoothing** - Reduces jitter without lag
4. **Morphology optimization** - Minimal iterations (1 each)
5. **ROI option** - Face region removal reduces processing area

## Project Structure

```
hand_boundary/
├── main.py                 # Main application loop
├── hand_detector.py        # HandDetector class
├── distance_utils.py       # Distance calibration utilities
├── config.py              # Configuration parameters
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── calibration.json      # Generated after calibration
```

## Configuration

Edit `config.py` to customize:

- Frame size (WIDTH/HEIGHT)
- HSV color ranges
- Distance thresholds (SAFE_PX, WARNING_PX, DANGER_PX)
- Virtual object radius
- Performance options (threading, background subtraction)

## Test Cases

The system has been tested for:

1. ✅ No hand detected → NO_HAND state
2. ✅ Hand inside circle → DANGER state + "DANGER DANGER" alert
3. ✅ Hand at intermediate distance → WARNING state
4. ✅ Hand far from circle → SAFE state
5. ✅ Performance: Average FPS ≥ 8 on modern laptop (i5+ processor)

## Technical Details

### Algorithm Summary

- **Color spaces**: HSV + YCrCb combined for robustness
- **Morphology kernel**: 5×5 ellipse
- **Contour method**: RETR_EXTERNAL with CHAIN_APPROX_SIMPLE
- **Distance method**: Euclidean distance from contour points to circle perimeter
- **Smoothing**: Exponential moving average (alpha = 0.7)

### Why This Approach?

- ✅ No restricted APIs (MediaPipe/OpenPose)
- ✅ Fast enough for real-time (8+ FPS on CPU)
- ✅ Robust to various skin tones with HSV+YCrCb
- ✅ Classical CV techniques as required by assignment
- ✅ Handles partial occlusions and varied orientations

## References

This implementation is based on classical computer vision techniques:

1. **Skin detection**: HSV color space segmentation
2. **Contour analysis**: OpenCV findContours + convex hull
3. **Distance estimation**: Triangle similarity method (optional)
4. **Performance**: Threaded capture and vectorized operations

## License

This is an educational prototype for internship interview purposes.

## Author

Created for Arvyax internship application - December 2025

## Submission Checklist

- [x] Real-time hand tracking without MediaPipe/OpenPose
- [x] Classical CV techniques (color segmentation, contours, convex hull)
- [x] Virtual boundary with distance-based states
- [x] SAFE/WARNING/DANGER classification
- [x] "DANGER DANGER" alert with blinking effect
- [x] ≥8 FPS on CPU-only execution
- [x] Calibration support (triangle similarity)
- [x] HSV trackbars for tuning
- [x] Complete documentation
- [x] Ready to run code

---

**Enjoy tracking! For questions or improvements, feel free to reach out.**
"""