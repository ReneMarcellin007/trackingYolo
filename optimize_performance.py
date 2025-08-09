#!/usr/bin/env python3
"""
Script d'optimisation des performances pour le tracking YOLOv7
Réduit la charge de calcul tout en maintenant la qualité
"""

import os

def create_optimized_tracker():
    """Crée une version optimisée du tracker pour meilleures performances"""
    
    optimized_code = '''import sys
sys.path.insert(0, '/jevoispro/config')
import pyjevois

if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
import math
from collections import deque

## YOLOv7 Optimized Tracker - Performance Edition
# Optimisations pour atteindre 30+ FPS
class PyPostYolov7ColorTrackerOptimized:
    
    def __init__(self):
        # Tracking state
        self.tracks = {}
        self.next_id = 1
        self.max_missing_frames = 20  # Réduit de 30
        
        # Detection results
        self.classIds = []
        self.confidences = []
        self.boxes = []
        self.classmap = None
        
        # C++ YOLO decoder
        self.yolopp = jevois.PyPostYOLO()
        
        # Frame counter pour traitement alterné
        self.frame_count = 0
        self.skip_color = False
        
    def init(self):
        """Paramètres optimisés pour performance"""
        pc = jevois.ParameterCategory("YOLOv7 Optimized Tracking", "")
        
        # Standard YOLO parameters
        self.classes = jevois.Parameter(self, 'classes', 'str',
                        "Path to text file with names of object classes",
                        '', pc)
        self.classes.setCallback(self.loadClasses)

        self.detecttype = jevois.Parameter(self, 'detecttype', 'str', 
                                           "Type of detection output format",
                                           'RAWYOLO', pc)
        
        # Seuils augmentés pour réduire le nombre de détections
        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                        "Classification threshold in percent",
                        30.0, pc)  # Augmenté de 20 à 30
        
        self.dthresh = jevois.Parameter(self, 'dthresh', 'float',
                        "Detection threshold in percent",
                        25.0, pc)  # Augmenté de 15 à 25

        self.nms = jevois.Parameter(self, 'nms', 'float',
                        "NMS threshold in percent",
                        50.0, pc)  # Augmenté de 45 à 50
                        
        self.maxnbox = jevois.Parameter(self, 'maxnbox', 'uint',
                        "Max number of boxes",
                        200, pc)  # Réduit de 500 à 200
                        
        self.sigmoid = jevois.Parameter(self, 'sigmoid', 'bool',
                        "Apply sigmoid",
                        False, pc)
                        
        self.classoffset = jevois.Parameter(self, 'classoffset', 'int',
                        "Class offset",
                        0, pc)
        
        # Tracking parameters optimisés
        self.max_distance = jevois.Parameter(self, 'max_distance', 'float',
                        "Maximum distance for tracking association",
                        150.0, pc)  # Réduit de 200 à 150
                        
        self.enable_tracking = jevois.Parameter(self, 'enable_tracking', 'bool',
                        "Enable object tracking with IDs",
                        True, pc)
        
        # Désactiver les features coûteuses par défaut
        self.use_color = jevois.Parameter(self, 'use_color', 'bool',
                        "Enable color-based tracking enhancement",
                        False, pc)  # Désactivé par défaut
                        
        self.enable_prediction = jevois.Parameter(self, 'enable_prediction', 'bool',
                        "Enable position prediction",
                        False, pc)  # Désactivé par défaut
    
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

    def simple_tracking_cost(self, detection, track):
        """Calcul de coût simplifié pour performance"""
        # Distance euclidienne simple
        d1 = detection['box']
        t1 = track['box']
        dist = math.sqrt((d1[0]+d1[2]-t1[0]-t1[2])**2/4 + (d1[1]+d1[3]-t1[1]-t1[3])**2/4)
        
        if dist > self.max_distance.get():
            return float('inf')
            
        # Pénalité si changement de classe
        if detection['class_id'] != track['class_id']:
            dist *= 2.0
            
        return dist / self.max_distance.get()

    def update_tracks_fast(self, new_boxes, new_classids, new_confs):
        """Version optimisée du tracking sans couleur ni prédiction"""
        if not self.enable_tracking.get():
            return new_boxes, new_classids, new_confs
            
        # Créer détections simples
        detections = []
        for box, class_id, conf in zip(new_boxes, new_classids, new_confs):
            detections.append({
                'box': box,
                'class_id': class_id,
                'confidence': conf
            })
        
        # Association rapide
        used_tracks = set()
        used_detections = set()
        
        for det_idx, detection in enumerate(detections):
            best_track = None
            best_cost = 1.5  # Seuil direct
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                    
                cost = self.simple_tracking_cost(detection, track)
                if cost < best_cost:
                    best_cost = cost
                    best_track = track_id
            
            if best_track is not None:
                # Mise à jour simple
                self.tracks[best_track].update({
                    'box': detection['box'],
                    'class_id': detection['class_id'],
                    'confidence': detection['confidence'],
                    'missing_frames': 0
                })
                used_tracks.add(best_track)
                used_detections.add(det_idx)
        
        # Nouveaux tracks
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detections:
                self.tracks[self.next_id] = {
                    'box': detection['box'],
                    'class_id': detection['class_id'],
                    'confidence': detection['confidence'],
                    'missing_frames': 0
                }
                self.next_id += 1
        
        # Suppression rapide des tracks perdus
        lost_tracks = []
        for track_id in self.tracks:
            if track_id not in used_tracks:
                self.tracks[track_id]['missing_frames'] += 1
                if self.tracks[track_id]['missing_frames'] > self.max_missing_frames:
                    lost_tracks.append(track_id)
        
        for track_id in lost_tracks:
            del self.tracks[track_id]
        
        # Retour simplifié
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
        """Processing optimisé"""
        if len(outs) == 0: 
            return
            
        # Clear results
        self.classIds.clear()
        self.confidences.clear()
        self.boxes.clear()
        
        # Frame counter
        self.frame_count += 1
        
        # Get image size
        self.imagew, self.imageh = preproc.imagesize()
        bw, bh = preproc.blobsize(0)

        try:
            # YOLO processing
            classids, confs, boxes = self.yolopp.yolo(outs,
                                                      len(self.classmap) if self.classmap else 80,
                                                      self.dthresh.get() * 0.01,
                                                      self.cthresh.get() * 0.01,
                                                      bw, bh,
                                                      self.classoffset.get(),
                                                      self.maxnbox.get(),
                                                      self.sigmoid.get())
            
            if len(boxes) > 0:
                # NMS avec seuil plus élevé
                indices = cv2.dnn.NMSBoxes(boxes, confs, 
                                           self.cthresh.get() * 0.01, 
                                           self.nms.get() * 0.01)

                final_boxes = []
                final_classids = []
                final_confs = []
                
                # Limiter le nombre de détections
                max_detections = min(20, len(indices))
                for i in range(max_detections):
                    idx = indices[i] if i < len(indices) else i
                    x, y, w, h = boxes[idx]
                    x1 = min(bw - 1, max(0, x))
                    x2 = min(bw - 1, max(0, x + w))
                    y1 = min(bh - 1, max(0, y))
                    y2 = min(bh - 1, max(0, y + h))

                    x1, y1 = preproc.b2i(x1, y1, 0)
                    x2, y2 = preproc.b2i(x2, y2, 0)
                    
                    final_boxes.append([x1, y1, x2, y2])
                    final_classids.append(classids[idx])
                    final_confs.append(confs[idx])
                
                # Tracking rapide
                self.boxes, self.classIds, self.confidences = self.update_tracks_fast(
                    final_boxes, final_classids, final_confs)
                    
        except Exception as e:
            jevois.LERROR(f"Processing failed: {e}")

    def getLabel(self, id, conf, track_id=None):
        """Label simplifié"""
        if self.classmap and id < len(self.classmap): 
            categ = self.classmap[id]
        else: 
            categ = 'object'
        
        if self.enable_tracking.get() and track_id is not None:
            return f"[{track_id}] {categ}: {conf:.0f}%"
        else:
            return f"{categ}: {conf:.0f}%"
    
    def report(self, outimg, helper, overlay, idle):
        """Report optimisé sans calculs lourds"""
        if len(self.boxes) == 0:
            return
        
        # Trouver les track IDs
        track_ids = {}
        if self.enable_tracking.get():
            for tid, track in self.tracks.items():
                for i, box in enumerate(self.boxes):
                    if box == track['box']:
                        track_ids[i] = tid
                        break
        
        # Affichage simple
        for i, (box, cls, conf) in enumerate(zip(self.boxes, self.classIds, self.confidences)):
            track_id = track_ids.get(i)
            label = self.getLabel(cls, conf * 100, track_id)
            
            x1, y1, x2, y2 = box
            
            # Couleur par classe
            if cls == 0:  # person
                col = 0xFF00FF00  # Vert
            elif cls in [1,2,3,5,7]:  # vehicles
                col = 0xFF00FFFF  # Cyan
            else:
                col = 0xFFFFFFFF  # Blanc
            
            if helper is not None:
                helper.drawRect(x1, y1, x2, y2, col, True)
                helper.drawText(x1 + 3, y1 + 3, label, col)
'''
    
    return optimized_code

# Créer la version optimisée
optimized_code = create_optimized_tracker()

# Sauvegarder
with open('/jevoispro/share/pydnn/post/PyPostYolov7ColorTrackerOptimized.py', 'w') as f:
    f.write(optimized_code)

# Créer aussi une version filtrée optimisée
filtered_optimized = optimized_code.replace(
    'class PyPostYolov7ColorTrackerOptimized:',
    'class PyPostYolov7ColorTrackerOptimizedFiltered:'
)

# Ajouter le filtre
lines = filtered_optimized.split('\n')
for i, line in enumerate(lines):
    if 'for i in range(max_detections):' in line:
        lines.insert(i+1, '                    idx = indices[i] if i < len(indices) else i')
        lines.insert(i+2, '                    # FILTER: Only keep person and vehicles')
        lines.insert(i+3, '                    if classids[idx] not in [0, 1, 2, 3, 5, 7]:')
        lines.insert(i+4, '                        continue')
        break

filtered_optimized = '\n'.join(lines)

with open('/jevoispro/share/pydnn/post/PyPostYolov7ColorTrackerOptimizedFiltered.py', 'w') as f:
    f.write(filtered_optimized)

print("✅ Versions optimisées créées!")
print("\nOptimisations appliquées:")
print("- Seuils de détection augmentés (30% / 25%)")
print("- NMS plus agressif (50%)")
print("- Maximum 200 boxes au lieu de 500")
print("- Maximum 20 détections finales")
print("- Tracking simplifié sans couleur ni prédiction")
print("- Distance max réduite à 150px")
print("\nObjectif: 30+ FPS avec tracking stable")