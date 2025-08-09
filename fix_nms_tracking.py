#!/usr/bin/env python3
"""
Fix pour le problème de détections multiples
Garde TOUTES les features mais corrige le NMS
"""

def create_fixed_tracker():
    """Version corrigée avec NMS approprié et Async"""
    
    fixed_code = '''import sys
sys.path.insert(0, '/jevoispro/config')
import pyjevois

if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
import math
from collections import deque

## YOLOv7 Tracker CORRIGÉ - Détections uniques comme OpenCV
class PyPostYolov7ColorTracker:
    
    def __init__(self):
        # Tracking state
        self.tracks = {}
        self.next_id = 1
        self.max_missing_frames = 30
        
        # Detection results
        self.classIds = []
        self.confidences = []
        self.boxes = []
        self.classmap = None
        
        # C++ YOLO decoder
        self.yolopp = jevois.PyPostYOLO()
        
        # Current input image for color extraction
        self.current_image = None
        
        # Context detection 
        self.context_history = deque(maxlen=30)
        self.current_context = "INDOOR"
        self.context_stable_frames = 0
        
    def init(self):
        """Parameters with all features enabled"""
        pc = jevois.ParameterCategory("YOLOv7 Fixed Tracking", "")
        
        # Standard YOLO parameters
        self.classes = jevois.Parameter(self, 'classes', 'str',
                        "Path to text file with names of object classes",
                        '', pc)
        self.classes.setCallback(self.loadClasses)

        self.detecttype = jevois.Parameter(self, 'detecttype', 'str', 
                                           "Type of detection output format",
                                           'RAWYOLO', pc)
        
        # CRITICAL: Higher thresholds like OpenCV model
        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                        "Classification threshold in percent",
                        50.0, pc)  # INCREASED to match OpenCV
        
        self.dthresh = jevois.Parameter(self, 'dthresh', 'float',
                        "Detection threshold in percent",
                        45.0, pc)  # INCREASED to match OpenCV

        self.nms = jevois.Parameter(self, 'nms', 'float',
                        "NMS threshold in percent",
                        45.0, pc)  # Standard NMS
                        
        self.maxnbox = jevois.Parameter(self, 'maxnbox', 'uint',
                        "Max number of boxes",
                        500, pc)
                        
        self.sigmoid = jevois.Parameter(self, 'sigmoid', 'bool',
                        "Apply sigmoid",
                        False, pc)
                        
        self.classoffset = jevois.Parameter(self, 'classoffset', 'int',
                        "Class offset",
                        0, pc)
        
        # Tracking parameters 
        self.max_distance = jevois.Parameter(self, 'max_distance', 'float',
                        "Maximum distance for tracking association",
                        200.0, pc)
                        
        self.enable_tracking = jevois.Parameter(self, 'enable_tracking', 'bool',
                        "Enable object tracking with IDs",
                        True, pc)
        
        # Keep all advanced features
        self.use_color = jevois.Parameter(self, 'use_color', 'bool',
                        "Enable color-based tracking enhancement",
                        True, pc)
                        
        self.color_weight = jevois.Parameter(self, 'color_weight', 'float',
                        "Weight of color similarity in tracking (0.0-1.0)",
                        0.3, pc)
                        
        self.min_saturation = jevois.Parameter(self, 'min_saturation', 'float',
                        "Minimum saturation for reliable color (0-255)",
                        40.0, pc)
                        
        self.min_brightness = jevois.Parameter(self, 'min_brightness', 'float',
                        "Minimum brightness for reliable color (0-255)",
                        60.0, pc)
        
        self.enable_prediction = jevois.Parameter(self, 'enable_prediction', 'bool',
                        "Enable position prediction",
                        True, pc)
                        
        self.enable_auto_context = jevois.Parameter(self, 'enable_auto_context', 'bool',
                        "Enable automatic INDOOR/CITY/HIGHWAY context detection",
                        True, pc)
    
    def freeze(self, doit):
        self.classes.freeze(doit)
        self.yolopp.freeze(doit)

    def loadClasses(self, filename):
        if filename:
            try:
                f = open(pyjevois.share + '/' + filename, 'rt')
                self.classmap = f.read().rstrip('\\n').split('\\n')
                f.close()
            except:
                self.classmap = []

    def predict_position(self, track):
        """Keep position prediction"""
        positions = track.get('position_history', [])
        if len(positions) < 2: return track['box'][:2]
        vx, vy = positions[-1][0] - positions[-2][0], positions[-1][1] - positions[-2][1]
        track['speed'] = math.sqrt(vx*vx + vy*vy)
        return [positions[-1][0] + vx, positions[-1][1] + vy]

    def detect_context(self, detections):
        """Keep context detection"""
        cars = sum(1 for d in detections if d['class_id'] in [2, 3, 5, 7])
        avg_speed = np.mean([track.get('speed', 0) for track in self.tracks.values()]) if self.tracks else 0
        new_context = "HIGHWAY" if cars >= 3 and avg_speed > 15 else "CITY" if cars >= 1 or avg_speed > 5 else "INDOOR"
        
        self.context_history.append(new_context)
        if len(self.context_history) >= 10:
            most_common = max(set(self.context_history), key=self.context_history.count)
            if most_common != self.current_context:
                if self.context_stable_frames > 15:
                    self.current_context = most_common
                    self.context_stable_frames = 0
                    self.update_context_parameters()
                else:
                    self.context_stable_frames += 1
            else:
                self.context_stable_frames = max(0, self.context_stable_frames - 1)

    def update_context_parameters(self):
        """Update tracking parameters based on context"""
        if self.current_context == "HIGHWAY":
            self.max_distance.set(200.0)
            self.max_missing_frames = 20
        elif self.current_context == "CITY":
            self.max_distance.set(150.0)
            self.max_missing_frames = 30
        else:  # INDOOR
            self.max_distance.set(120.0)
            self.max_missing_frames = 30

    def extract_color_info(self, img, box):
        """Keep color extraction"""
        if not self.use_color.get() or img is None:
            return {
                'h': 0, 's': 0, 'v': 128,
                'name': 'unknown',
                'bgr': (128, 128, 128),
                'valid': False
            }
            
        try:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            h, w = img.shape[:2]
            
            x1 = max(0, min(w-1, x1))
            x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1))
            y2 = max(0, min(h-1, y2))
            
            if x2 <= x1 or y2 <= y1:
                return {'h': 0, 's': 0, 'v': 128, 'name': 'invalid', 'bgr': (128, 128, 128), 'valid': False}
            
            margin = 0.15
            cx1 = int(x1 + (x2 - x1) * margin)
            cx2 = int(x2 - (x2 - x1) * margin)  
            cy1 = int(y1 + (y2 - y1) * margin)
            cy2 = int(y2 - (y2 - y1) * margin)
            
            roi = img[cy1:cy2:3, cx1:cx2:3]
            if roi.size == 0:
                return {'h': 0, 's': 0, 'v': 128, 'name': 'empty', 'bgr': (128, 128, 128), 'valid': False}
                
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv_roi)[:3]
            h_val, s_val, v_val = mean_hsv
            
            valid = (s_val >= self.min_saturation.get() and v_val >= self.min_brightness.get())
            
            color_name = self.hsv_to_name(h_val, s_val, v_val)
            color_bgr = self.hsv_to_bgr(h_val, s_val, v_val)
            
            return {
                'h': h_val, 's': s_val, 'v': v_val,
                'name': color_name,
                'bgr': color_bgr,
                'valid': valid
            }
            
        except Exception as e:
            return {'h': 0, 's': 0, 'v': 128, 'name': 'error', 'bgr': (128, 128, 128), 'valid': False}

    def hsv_to_name(self, h, s, v):
        """Convert HSV to readable color name"""
        if s < self.min_saturation.get():
            if v < 70: return "black"
            elif v > 200: return "white"  
            else: return "gray"
        
        if v < self.min_brightness.get():
            return "dark"
        
        if h < 10 or h >= 170: return "red"
        elif h < 25: return "orange"
        elif h < 35: return "yellow"
        elif h < 45: return "lime"
        elif h < 85: return "green"
        elif h < 100: return "cyan"
        elif h < 125: return "blue"
        elif h < 155: return "purple"
        else: return "magenta"

    def hsv_to_bgr(self, h, s, v):
        """Convert HSV to BGR for visualization"""
        try:
            hsv_pixel = np.uint8([[[h, s, v]]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            return tuple(map(int, bgr_pixel[0][0]))
        except:
            return (128, 128, 128)

    def color_similarity(self, color1, color2):
        """Calculate color similarity"""
        if not color1['valid'] or not color2['valid']:
            return 0.5
        
        h_diff = abs(color1['h'] - color2['h'])
        h_diff = min(h_diff, 360 - h_diff)
        
        s_diff = abs(color1['s'] - color2['s'])
        v_diff = abs(color1['v'] - color2['v'])
        
        similarity = 1.0 - (h_diff/180 * 0.6 + s_diff/255 * 0.3 + v_diff/255 * 0.1)
        return max(0.0, min(1.0, similarity))

    def distance_between_boxes(self, box1, box2):
        """Calculate distance between box centers"""
        c1_x = (box1[0] + box1[2]) / 2
        c1_y = (box1[1] + box1[3]) / 2
        c2_x = (box2[0] + box2[2]) / 2
        c2_y = (box2[1] + box2[3]) / 2
        return math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

    def distance_to_prediction(self, detection_box, predicted_pos):
        """Calculate distance to predicted position"""
        det_x = (detection_box[0] + detection_box[2]) / 2
        det_y = (detection_box[1] + detection_box[3]) / 2
        return math.sqrt((det_x - predicted_pos[0])**2 + (det_y - predicted_pos[1])**2)

    def tracking_cost(self, detection, track):
        """Keep full tracking cost with all features"""
        distance = self.distance_between_boxes(detection['box'], track['box'])
        if distance > self.max_distance.get():
            return float('inf')
        distance_cost = distance / self.max_distance.get()
        
        prediction_cost = 0.5
        if self.enable_prediction.get():
            predicted_pos = self.predict_position(track)
            pred_distance = self.distance_to_prediction(detection['box'], predicted_pos)
            prediction_cost = pred_distance / self.max_distance.get()
        
        color_cost = 0.5
        if self.use_color.get():
            color_sim = self.color_similarity(detection['color'], track['color'])
            color_cost = 1.0 - color_sim
        
        class_cost = 0.0 if detection['class_id'] == track['class_id'] else 0.8
        
        color_w = self.color_weight.get() if self.use_color.get() else 0.0
        pred_w = 0.3 if self.enable_prediction.get() else 0.0
        
        total_cost = (distance_cost * (1.0 - color_w - pred_w) + 
                     color_cost * color_w + 
                     prediction_cost * pred_w +
                     class_cost * 0.2)
        
        return total_cost

    def update_tracks(self, new_boxes, new_classids, new_confs):
        """Keep full tracking with all features"""
        if not self.enable_tracking.get():
            return new_boxes, new_classids, new_confs
            
        detections = []
        for box, class_id, conf in zip(new_boxes, new_classids, new_confs):
            color_info = self.extract_color_info(self.current_image, box)
            detections.append({
                'box': box,
                'class_id': class_id,
                'confidence': conf,
                'color': color_info
            })
        
        if self.enable_auto_context.get():
            self.detect_context(detections)
        
        used_tracks = set()
        used_detections = set()
        
        sorted_detections = sorted(enumerate(detections), key=lambda x: x[1]['confidence'], reverse=True)
        
        for det_idx, detection in sorted_detections:
            if det_idx in used_detections:
                continue
                
            best_track = None
            best_cost = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                    
                cost = self.tracking_cost(detection, track)
                if cost < best_cost:
                    best_cost = cost
                    best_track = track_id
            
            if best_track is not None and best_cost < 1.8:
                center_x = (detection['box'][0] + detection['box'][2]) / 2
                center_y = (detection['box'][1] + detection['box'][3]) / 2
                
                if 'position_history' not in self.tracks[best_track]:
                    self.tracks[best_track]['position_history'] = deque(maxlen=5)
                self.tracks[best_track]['position_history'].append((center_x, center_y))
                
                self.tracks[best_track].update({
                    'box': detection['box'],
                    'class_id': detection['class_id'],
                    'confidence': detection['confidence'],
                    'color': detection['color'],
                    'missing_frames': 0,
                    'tracking_quality': 1.0 - best_cost
                })
                used_tracks.add(best_track)
                used_detections.add(det_idx)
        
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detections:
                center_x = (detection['box'][0] + detection['box'][2]) / 2
                center_y = (detection['box'][1] + detection['box'][3]) / 2
                
                self.tracks[self.next_id] = {
                    'box': detection['box'],
                    'class_id': detection['class_id'],
                    'confidence': detection['confidence'],
                    'color': detection['color'],
                    'missing_frames': 0,
                    'tracking_quality': 1.0,
                    'position_history': deque([(center_x, center_y)], maxlen=5),
                    'speed': 0
                }
                self.next_id += 1
        
        lost_tracks = []
        for track_id in self.tracks:
            if track_id not in used_tracks:
                self.tracks[track_id]['missing_frames'] += 1
                if self.tracks[track_id]['missing_frames'] > self.max_missing_frames:
                    lost_tracks.append(track_id)
        
        for track_id in lost_tracks:
            del self.tracks[track_id]
        
        if len(self.tracks) > 0:
            tracked_boxes = []
            tracked_classids = []
            tracked_confs = []
            
            for track in self.tracks.values():
                tracked_boxes.append(track['box'])
                tracked_classids.append(track['class_id'])
                tracked_confs.append(track['confidence'])
                
            return tracked_boxes, tracked_classids, tracked_confs
        else:
            return new_boxes, new_classids, new_confs

    def process(self, outs, preproc):
        """FIXED: Proper NMS to avoid multiple detections"""
        if len(outs) == 0: 
            return
            
        # Clear results
        self.classIds.clear()
        self.confidences.clear()
        self.boxes.clear()
        
        # Get current image for color extraction
        self.current_image = None
        try:
            if hasattr(preproc, 'getImage'):
                self.current_image = preproc.getImage()
        except:
            pass
        
        # Get image size
        self.imagew, self.imageh = preproc.imagesize()
        bw, bh = preproc.blobsize(0)

        try:
            # YOLO processing with HIGHER thresholds like OpenCV
            classids, confs, boxes = self.yolopp.yolo(outs,
                                                      len(self.classmap) if self.classmap else 80,
                                                      self.dthresh.get() * 0.01,  # 45% threshold
                                                      self.cthresh.get() * 0.01,  # 50% threshold
                                                      bw, bh,
                                                      self.classoffset.get(),
                                                      self.maxnbox.get(),
                                                      self.sigmoid.get())
            
            if len(boxes) > 0:
                # CRITICAL FIX: Proper NMS to avoid multiple boxes
                # Convert to proper format for NMS
                nms_boxes = []
                nms_scores = []
                nms_classes = []
                
                for i in range(len(boxes)):
                    x, y, w, h = boxes[i]
                    # Ensure proper box format
                    nms_boxes.append([x, y, w, h])
                    nms_scores.append(float(confs[i]))
                    nms_classes.append(int(classids[i]))
                
                # Apply NMS properly
                indices = cv2.dnn.NMSBoxes(
                    nms_boxes, 
                    nms_scores, 
                    self.cthresh.get() * 0.01,  # Score threshold
                    self.nms.get() * 0.01        # NMS threshold
                )
                
                # Process NMS results correctly
                final_boxes = []
                final_classids = []
                final_confs = []
                
                if len(indices) > 0:
                    # Handle both formats of cv2.dnn.NMSBoxes output
                    if isinstance(indices, tuple):
                        indices = indices[0]
                    indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                    
                    for idx in indices:
                        x, y, w, h = nms_boxes[idx]
                        
                        # Convert to image coordinates
                        x1 = int(x * self.imagew / bw)
                        y1 = int(y * self.imageh / bh)
                        x2 = int((x + w) * self.imagew / bw)
                        y2 = int((y + h) * self.imageh / bh)
                        
                        # Clamp to image bounds
                        x1 = max(0, min(self.imagew - 1, x1))
                        y1 = max(0, min(self.imageh - 1, y1))
                        x2 = max(0, min(self.imagew - 1, x2))
                        y2 = max(0, min(self.imageh - 1, y2))
                        
                        if x2 > x1 and y2 > y1:  # Valid box
                            final_boxes.append([x1, y1, x2, y2])
                            final_classids.append(nms_classes[idx])
                            final_confs.append(nms_scores[idx])
                
                # Apply tracking with all features
                self.boxes, self.classIds, self.confidences = self.update_tracks(
                    final_boxes, final_classids, final_confs)
                    
        except Exception as e:
            jevois.LERROR(f"Processing failed: {e}")

    def getLabel(self, id, conf, track_id=None, color_info=None):
        """Get label with all info"""
        if self.classmap is None or id < 0 or id >= len(self.classmap): 
            categ = 'object'
        else: 
            categ = self.classmap[id]
        
        if self.enable_tracking.get() and track_id is not None:
            label = f"ID:{track_id} {categ}: {conf * 100.0:.1f}%"
        else:
            label = f"{categ}: {conf * 100.0:.1f}%"
        
        if self.use_color.get() and color_info and color_info['valid']:
            label += f" ({color_info['name']})"
        
        if self.enable_auto_context.get():
            label += f" [{self.current_context}]"
        
        if color_info and color_info['valid']:
            color_bgr = color_info['bgr']
            color = ((color_bgr[2] & 0xFF) << 24) | ((color_bgr[1] & 0xFF) << 16) | ((color_bgr[0] & 0xFF) << 8) | 0xFF
        else:
            color = jevois.stringToRGBA(categ, 255) & 0xffffffff
            
        return (label, color)
    
    def report(self, outimg, helper, overlay, idle):
        """Display with all features"""
        if helper is not None and overlay:
            # Status display
            if self.enable_auto_context.get():
                status = f"Fixed Tracking [{self.current_context}] ({len(self.tracks)} tracked)"
            else:
                status = f"Fixed Tracking ({len(self.tracks)} tracked)"
            helper.drawText(10, 30, status, 0xFFFFFFFF)
            
            for i, (class_id, conf) in enumerate(zip(self.classIds, self.confidences)):
                if i < len(self.boxes):
                    track_id = None
                    color_info = None
                    quality = 1.0
                    speed = 0
                    if self.enable_tracking.get():
                        current_box = self.boxes[i]
                        for tid, track in self.tracks.items():
                            if track['box'] == current_box:
                                track_id = tid
                                color_info = track.get('color')
                                quality = track.get('tracking_quality', 1.0)
                                speed = track.get('speed', 0)
                                break
                    
                    label, color = self.getLabel(class_id, conf, track_id, color_info)
                    
                    if track_id is not None:
                        if quality > 0.8: label += " ★★★"
                        elif quality > 0.6: label += " ★★"
                        elif quality > 0.4: label += " ★"
                    
                    x1, y1, x2, y2 = self.boxes[i]
                    helper.drawRect(x1, y1, x2, y2, color, True)
                    helper.drawText(x1 + 3, y1 + 3, label, color)
'''
    
    return fixed_code

# Create fixed version
print("Creating fixed tracker with proper NMS...")
fixed_code = create_fixed_tracker()

# Save it
with open('/jevoispro/share/pydnn/post/PyPostYolov7ColorTrackerFixed.py', 'w') as f:
    f.write(fixed_code)

print("✅ Fixed tracker created!")

# Also create filtered version
filtered_code = fixed_code.replace(
    'class PyPostYolov7ColorTracker:',
    'class PyPostYolov7ColorTrackerFilteredFixed:'
).replace(
    '# Process NMS results correctly',
    '''# Process NMS results correctly
                # FILTER: Only persons and vehicles'''
).replace(
    'for idx in indices:',
    '''for idx in indices:
                        # Filter classes
                        if nms_classes[idx] not in [0, 1, 2, 3, 5, 7]:
                            continue'''
)

with open('/jevoispro/share/pydnn/post/PyPostYolov7ColorTrackerFilteredFixed.py', 'w') as f:
    f.write(filtered_code)

print("✅ Fixed filtered tracker created!")
print("\nCe qui a été corrigé:")
print("1. NMS approprié pour éviter les détections multiples")
print("2. Seuils augmentés à 45-50% comme OpenCV")
print("3. Conversion correcte des coordonnées blob->image")
print("4. TOUTES les features gardées (couleur, prédiction, contexte)")
print("5. Mode Async disponible dans la config YAML")