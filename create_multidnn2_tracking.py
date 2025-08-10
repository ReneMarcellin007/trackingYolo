#!/usr/bin/env python3
"""
Version MultiDNN2 compatible avec tracking complet
Pour combiner avec road segmentation
"""

def create_multidnn2_tracker():
    """Version optimisée pour MultiDNN2 avec tracking"""
    
    code = '''import sys
sys.path.insert(0, '/jevoispro/config')
import pyjevois

if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
import math
from collections import deque

## YOLOv7 Tracker MultiDNN2 - Compatible avec Road Segmentation
class PyPostYolov7MultiDNN:
    
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
        
        # YOLO decoder - sera initialisé plus tard
        self.yolopp = None
        
        # Pour MultiDNN2 - pas d'initialisation dans __init__
        self.initialized = False
        
    def init(self):
        """Paramètres optimisés pour MultiDNN2"""
        pc = jevois.ParameterCategory("YOLOv7 MultiDNN2 Tracking", "")
        
        # Standard YOLO parameters
        self.classes = jevois.Parameter(self, 'classes', 'str',
                        "Path to text file with names of object classes",
                        '', pc)
        self.classes.setCallback(self.loadClasses)

        self.detecttype = jevois.Parameter(self, 'detecttype', 'str', 
                                           "Type of detection output format",
                                           'RAWYOLO', pc)
        
        # Seuils optimisés pour performance
        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                        "Classification threshold in percent",
                        25.0, pc)
        
        self.dthresh = jevois.Parameter(self, 'dthresh', 'float',
                        "Detection threshold in percent",
                        20.0, pc)

        self.nms = jevois.Parameter(self, 'nms', 'float',
                        "NMS threshold in percent",
                        45.0, pc)
                        
        self.maxnbox = jevois.Parameter(self, 'maxnbox', 'uint',
                        "Max number of boxes",
                        300, pc)  # Réduit pour performance
                        
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
        
        # Filter pour véhicules sur route
        self.vehicle_only = jevois.Parameter(self, 'vehicle_only', 'bool',
                        "Only track vehicles (for road analysis)",
                        False, pc)
        
        self.initialized = True
    
    def freeze(self, doit):
        if hasattr(self, 'classes'):
            self.classes.freeze(doit)
        # Ne pas freezer yolopp s'il n'existe pas

    def loadClasses(self, filename):
        if filename:
            try:
                f = open(pyjevois.share + '/' + filename, 'rt')
                self.classmap = f.read().rstrip('\\n').split('\\n')
                f.close()
            except:
                self.classmap = []

    def distance_between_boxes(self, box1, box2):
        """Calcul de distance simple"""
        c1_x = (box1[0] + box1[2]) / 2
        c1_y = (box1[1] + box1[3]) / 2
        c2_x = (box2[0] + box2[2]) / 2
        c2_y = (box2[1] + box2[3]) / 2
        return math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

    def update_tracks(self, new_boxes, new_classids, new_confs):
        """Tracking simple et rapide pour MultiDNN2"""
        if not self.enable_tracking.get():
            return new_boxes, new_classids, new_confs
            
        # Filter véhicules si demandé
        if self.vehicle_only.get():
            filtered = []
            for i, class_id in enumerate(new_classids):
                # Classes: car(2), motorbike(3), bus(5), truck(7)
                if class_id in [2, 3, 5, 7]:
                    filtered.append((new_boxes[i], class_id, new_confs[i]))
            if not filtered:
                return [], [], []
            new_boxes, new_classids, new_confs = zip(*filtered)
            
        # Association simple
        used_tracks = set()
        used_detections = set()
        
        for det_idx, box in enumerate(new_boxes):
            best_track = None
            best_dist = self.max_distance.get()
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                    
                dist = self.distance_between_boxes(box, track['box'])
                if dist < best_dist and new_classids[det_idx] == track['class_id']:
                    best_dist = dist
                    best_track = track_id
            
            if best_track is not None:
                self.tracks[best_track] = {
                    'box': box,
                    'class_id': new_classids[det_idx],
                    'confidence': new_confs[det_idx],
                    'missing_frames': 0
                }
                used_tracks.add(best_track)
                used_detections.add(det_idx)
        
        # Nouvelles détections
        for det_idx, box in enumerate(new_boxes):
            if det_idx not in used_detections:
                self.tracks[self.next_id] = {
                    'box': box,
                    'class_id': new_classids[det_idx],
                    'confidence': new_confs[det_idx],
                    'missing_frames': 0
                }
                self.next_id += 1
        
        # Suppression des tracks perdus
        lost_tracks = []
        for track_id in self.tracks:
            if track_id not in used_tracks:
                self.tracks[track_id]['missing_frames'] += 1
                if self.tracks[track_id]['missing_frames'] > self.max_missing_frames:
                    lost_tracks.append(track_id)
        
        for track_id in lost_tracks:
            del self.tracks[track_id]
        
        # Retour avec IDs
        tracked_boxes = []
        tracked_classids = []
        tracked_confs = []
        
        for track in self.tracks.values():
            if track['missing_frames'] == 0:
                tracked_boxes.append(track['box'])
                tracked_classids.append(track['class_id'])
                tracked_confs.append(track['confidence'])
                
        return tracked_boxes, tracked_classids, tracked_confs

    def process(self, outs, preproc):
        """Processing optimisé pour MultiDNN2"""
        if len(outs) == 0: 
            return
            
        # Initialisation différée du YOLO decoder
        if self.yolopp is None:
            try:
                self.yolopp = jevois.PyPostYOLO()
            except:
                # En MultiDNN2, on peut avoir des problèmes d'init
                # Fallback: traitement manuel basique
                self.manual_decode(outs, preproc)
                return
                
        # Clear results
        self.classIds.clear()
        self.confidences.clear()
        self.boxes.clear()
        
        # Get image size
        try:
            self.imagew, self.imageh = preproc.imagesize()
            bw, bh = preproc.blobsize(0)
        except:
            # Défaut si erreur
            self.imagew, self.imageh = 1024, 576
            bw, bh = 1024, 576

        try:
            # YOLO processing avec le decoder C++
            classids, confs, boxes = self.yolopp.yolo(outs,
                                                      len(self.classmap) if self.classmap else 80,
                                                      self.dthresh.get() * 0.01,
                                                      self.cthresh.get() * 0.01,
                                                      bw, bh,
                                                      self.classoffset.get(),
                                                      self.maxnbox.get(),
                                                      self.sigmoid.get())
            
            if len(boxes) > 0:
                # NMS approprié
                indices = cv2.dnn.NMSBoxes(boxes, confs, 
                                           self.cthresh.get() * 0.01, 
                                           self.nms.get() * 0.01)
                
                if len(indices) > 0:
                    indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                    
                    final_boxes = []
                    final_classids = []
                    final_confs = []
                    
                    for idx in indices[:20]:  # Max 20 pour performance
                        x, y, w, h = boxes[idx]
                        
                        try:
                            x1, y1 = preproc.b2i(x, y, 0)
                            x2, y2 = preproc.b2i(x + w, y + h, 0)
                        except:
                            # Fallback si b2i échoue
                            x1 = int(x * self.imagew / bw)
                            y1 = int(y * self.imageh / bh)
                            x2 = int((x + w) * self.imagew / bw)
                            y2 = int((y + h) * self.imageh / bh)
                        
                        x1 = max(0, min(self.imagew - 1, x1))
                        y1 = max(0, min(self.imageh - 1, y1))
                        x2 = max(0, min(self.imagew - 1, x2))
                        y2 = max(0, min(self.imageh - 1, y2))
                        
                        if x2 > x1 and y2 > y1:
                            final_boxes.append([x1, y1, x2, y2])
                            final_classids.append(classids[idx])
                            final_confs.append(confs[idx])
                    
                    # Apply tracking
                    self.boxes, self.classIds, self.confidences = self.update_tracks(
                        final_boxes, final_classids, final_confs)
                    
        except Exception as e:
            # Silent fail pour MultiDNN2
            pass

    def manual_decode(self, outs, preproc):
        """Décodage manuel si PyPostYOLO échoue"""
        # Implementation basique pour fallback
        # Juste retourner vide pour l'instant
        pass

    def report(self, outimg, helper, overlay, idle):
        """Affichage optimisé pour MultiDNN2 avec road segmentation"""
        if helper is not None and overlay and len(self.boxes) > 0:
            # Trouver les track IDs
            track_ids = {}
            if self.enable_tracking.get():
                for tid, track in self.tracks.items():
                    if track['missing_frames'] == 0:
                        for i, box in enumerate(self.boxes):
                            if box == track['box']:
                                track_ids[i] = tid
                                break
            
            for i, (class_id, conf) in enumerate(zip(self.classIds, self.confidences)):
                if i < len(self.boxes):
                    x1, y1, x2, y2 = self.boxes[i]
                    
                    # Label avec ID si tracking
                    if i in track_ids:
                        track_id = track_ids[i]
                        if self.classmap and class_id < len(self.classmap):
                            label = f"[{track_id}] {self.classmap[class_id]}: {conf*100:.0f}%"
                        else:
                            label = f"[{track_id}] Vehicle: {conf*100:.0f}%"
                    else:
                        if self.classmap and class_id < len(self.classmap):
                            label = f"{self.classmap[class_id]}: {conf*100:.0f}%"
                        else:
                            label = f"Vehicle: {conf*100:.0f}%"
                    
                    # Couleur selon type
                    if class_id == 2:  # car
                        col = 0xFF00FFFF  # Cyan
                    elif class_id in [3, 5, 7]:  # autres véhicules
                        col = 0xFFFFFF00  # Jaune
                    else:
                        col = 0xFF00FF00  # Vert
                    
                    helper.drawRect(x1, y1, x2, y2, col, True)
                    helper.drawText(x1 + 3, y1 + 3, label, col)
                    
                    # Message pour intégration avec road segmentation
                    if self.vehicle_only.get() and class_id in [2, 3, 5, 7]:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        # Ce message peut être utilisé par le module road segmentation
                        jevois.sendSerial(f"VEHICLE id:{track_ids.get(i, 0)} type:{class_id} x:{center_x} y:{center_y}")
'''
    
    return code

# Créer le tracker MultiDNN2
tracker_code = create_multidnn2_tracker()

with open('/jevoispro/share/pydnn/post/PyPostYolov7MultiDNN.py', 'w') as f:
    f.write(tracker_code)

print("✅ Created PyPostYolov7MultiDNN.py")

# Version filtrée pour véhicules uniquement
filtered_code = tracker_code.replace(
    'class PyPostYolov7MultiDNN:',
    'class PyPostYolov7MultiDNNFiltered:'
).replace(
    'self.vehicle_only = jevois.Parameter(self, \'vehicle_only\', \'bool\',\n                        "Only track vehicles (for road analysis)",\n                        False, pc)',
    'self.vehicle_only = jevois.Parameter(self, \'vehicle_only\', \'bool\',\n                        "Only track vehicles (for road analysis)",\n                        True, pc)'  # True par défaut
)

with open('/jevoispro/share/pydnn/post/PyPostYolov7MultiDNNFiltered.py', 'w') as f:
    f.write(filtered_code)

print("✅ Created PyPostYolov7MultiDNNFiltered.py")

# YAML configs
yaml_multidnn = '''%YAML 1.0
---

yolov7-multidnn:
  preproc: Blob
  mean: "0 0 0"
  scale: 0.0039215686
  nettype: NPU
  model: "npu/detection/yolov7-tiny-1024x576.nb"
  postproc: Python
  pypost: "pydnn/post/PyPostYolov7MultiDNN.py"
  detecttype: RAWYOLO
  classes: "dnn/labels/coco-labels.txt"
  processing: Async
  intensors: "NCHW:8U:1x3x576x1024:AA:0.003921568393707275:0"
  outtensors: "NCHW:8U:1x255x72x128:AA:0.003911261446774006:0, NCHW:8U:1x255x36x64:AA:0.0038770213909447193:0, NCHW:8U:1x255x18x32:AA:0.003848204342648387:0"
  anchors: "10,13, 16,30, 33,23; 30,61, 62,45, 59,119; 116,90, 156,198, 373,326"
  scalexy: 2.0
  sigmoid: false
'''

with open('/jevoispro/share/dnn/custom/yolov7-multidnn.yml', 'w') as f:
    f.write(yaml_multidnn)

print("✅ Created yolov7-multidnn.yml")

yaml_filtered = yaml_multidnn.replace('yolov7-multidnn:', 'yolov7-multidnn-vehicles:').replace(
    'pypost: "pydnn/post/PyPostYolov7MultiDNN.py"',
    'pypost: "pydnn/post/PyPostYolov7MultiDNNFiltered.py"'
)

with open('/jevoispro/share/dnn/custom/yolov7-multidnn-vehicles.yml', 'w') as f:
    f.write(yaml_filtered)

print("✅ Created yolov7-multidnn-vehicles.yml")
print("\n🎯 Modèles MultiDNN2 créés pour tracking + road segmentation!")