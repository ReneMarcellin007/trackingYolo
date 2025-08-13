#!/usr/bin/env python3
"""
🎯 YOLO ULTIMATE - Le SEUL modèle qui fonctionne
Basé sur YOLOv7 - Compatible DNN & MultiDNN2
IDs aléatoires + Tracking persistant
"""

import numpy as np
import random
import time
from collections import deque

class PyPostYOLO_Ultimate:
    """Le post-processeur YOLO définitif - Simple et efficace"""
    
    def __init__(self):
        # Configuration YOLOv7-tiny UNIQUEMENT
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        
        # Anchors YOLOv7-tiny (TESTÉ ET VALIDÉ)
        self.anchors = [
            [(10, 13), (16, 30), (33, 23)],      # P3/8
            [(30, 61), (62, 45), (59, 119)],     # P4/16
            [(116, 90), (156, 198), (373, 326)]  # P5/32
        ]
        self.strides = [8, 16, 32]
        self.scale_xy = 2.0
        
        # Tracking simple mais efficace
        self.tracks = {}
        self.next_id = 1
        
        # Classes COCO
        self.classmap = self._load_classes()
        
        # Optimisation simple
        self.grid_cache = {}
        
    def _load_classes(self):
        """Charge les classes COCO"""
        try:
            with open('/jevoispro/share/dnn/labels/coco-labels.txt', 'r') as f:
                return f.read().strip().split('\n')
        except:
            return [f"class{i}" for i in range(80)]
    
    def init(self):
        """Initialisation JeVois"""
        pass
    
    def process(self, outs, preproc):
        """Process principal - YOLOv7 uniquement"""
        
        # Dimensions de l'image
        try:
            bsiz = preproc.blobsize(0)
            img_w, img_h = bsiz[1], bsiz[0]
        except:
            img_w, img_h = 512, 288
        
        # Décoder les 3 échelles YOLOv7
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for scale_idx in range(min(3, len(outs))):
            boxes, scores, classes = self._decode_yolov7(
                outs[scale_idx], 
                scale_idx,
                img_w, 
                img_h
            )
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_classes.extend(classes)
        
        # NMS simple
        final_indices = self._nms(all_boxes, all_scores)
        
        # Créer les détections finales
        detections = []
        for idx in final_indices:
            det = {
                'box': all_boxes[idx],
                'score': all_scores[idx],
                'class_id': all_classes[idx],
                'class_name': self.classmap[all_classes[idx]] if all_classes[idx] < len(self.classmap) else f"class{all_classes[idx]}"
            }
            
            # Ajouter tracking
            det['id'] = self._get_track_id(det)
            det['random_id'] = random.randint(100, 999)
            
            detections.append(det)
        
        # Stocker pour report
        self.detections = detections
        return detections
    
    def _decode_yolov7(self, output, scale_idx, img_w, img_h):
        """Décode une échelle YOLOv7"""
        
        # Format attendu: [1, 255, H, W]
        if len(output.shape) != 4 or output.shape[1] != 255:
            return [], [], []
        
        _, _, grid_h, grid_w = output.shape
        stride = self.strides[scale_idx]
        anchors = self.anchors[scale_idx]
        
        # Reshape: [1, 255, H, W] -> [3, 85, H, W]
        output = output.reshape(3, 85, grid_h, grid_w)
        
        # Créer ou récupérer la grille
        grid_key = (grid_w, grid_h)
        if grid_key not in self.grid_cache:
            xv, yv = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            self.grid_cache[grid_key] = (xv, yv)
        xv, yv = self.grid_cache[grid_key]
        
        boxes = []
        scores = []
        classes = []
        
        # Process chaque anchor
        for a in range(3):
            anchor_w, anchor_h = anchors[a]
            
            # Extraire les prédictions pour cet anchor
            pred = output[a]  # [85, H, W]
            
            # Objectness
            obj_conf = self._sigmoid(pred[4])
            
            # Masque des cellules valides
            valid = obj_conf > self.conf_threshold
            if not np.any(valid):
                continue
            
            # Coordonnées des cellules valides
            valid_y, valid_x = np.where(valid)
            
            # Process seulement les cellules valides
            for i in range(len(valid_x)):
                x_idx = valid_x[i]
                y_idx = valid_y[i]
                
                # Prédictions pour cette cellule
                cell_pred = pred[:, y_idx, x_idx]
                
                # Coordonnées de la boîte
                tx = self._sigmoid(cell_pred[0])
                ty = self._sigmoid(cell_pred[1])
                tw = cell_pred[2]
                th = cell_pred[3]
                
                # Décoder position (formule YOLOv7)
                bx = (tx * self.scale_xy - 0.5 * (self.scale_xy - 1) + x_idx) * stride
                by = (ty * self.scale_xy - 0.5 * (self.scale_xy - 1) + y_idx) * stride
                
                # Décoder taille
                bw = np.exp(np.clip(tw, -5, 5)) * anchor_w
                bh = np.exp(np.clip(th, -5, 5)) * anchor_h
                
                # Classes
                class_probs = self._sigmoid(cell_pred[5:85])
                class_id = np.argmax(class_probs)
                class_conf = class_probs[class_id]
                
                # Score final
                score = obj_conf[y_idx, x_idx] * class_conf
                
                if score > self.conf_threshold:
                    # Convertir en format [x1, y1, x2, y2]
                    x1 = (bx - bw/2) / img_w
                    y1 = (by - bh/2) / img_h
                    x2 = (bx + bw/2) / img_w
                    y2 = (by + bh/2) / img_h
                    
                    # Clamp to [0, 1]
                    x1 = max(0, min(1, x1))
                    y1 = max(0, min(1, y1))
                    x2 = max(0, min(1, x2))
                    y2 = max(0, min(1, y2))
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(score))
                    classes.append(int(class_id))
        
        return boxes, scores, classes
    
    def _sigmoid(self, x):
        """Sigmoid avec protection overflow"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def _nms(self, boxes, scores, threshold=0.45):
        """Non-Maximum Suppression simple"""
        if not boxes:
            return []
        
        # Convertir en numpy pour efficacité
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Trier par score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            # Prendre le meilleur
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculer IoU avec les autres
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            iou = inter / (area_i + area - inter)
            
            # Garder seulement ceux avec IoU < threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _get_track_id(self, detection):
        """Tracking simple par position"""
        
        # Position du centre
        cx = (detection['box'][0] + detection['box'][2]) / 2
        cy = (detection['box'][1] + detection['box'][3]) / 2
        
        # Chercher un track proche
        best_track = None
        best_dist = 0.05  # 5% de l'image max
        
        for track_id, track in self.tracks.items():
            if track['class_id'] != detection['class_id']:
                continue
            
            # Distance au track
            dist = np.sqrt((cx - track['cx'])**2 + (cy - track['cy'])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_track = track_id
        
        # Si trouvé, mettre à jour
        if best_track:
            self.tracks[best_track]['cx'] = cx
            self.tracks[best_track]['cy'] = cy
            self.tracks[best_track]['time'] = time.time()
            return best_track
        
        # Sinon créer nouveau track
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'cx': cx,
            'cy': cy,
            'class_id': detection['class_id'],
            'time': time.time()
        }
        
        # Nettoyer vieux tracks (>2 secondes)
        current = time.time()
        self.tracks = {k: v for k, v in self.tracks.items() 
                      if current - v['time'] < 2.0}
        
        return track_id
    
    def report(self, outimg, helper, overlay, idle):
        """Affichage des résultats"""
        
        if hasattr(self, 'detections'):
            for det in self.detections:
                # Format: ID_track/ID_random: class score%
                label = f"ID{det['id']}/{det['random_id']}:{det['class_name']} {det['score']*100:.1f}%"
                print(label)
        
        return len(self.detections) if hasattr(self, 'detections') else 0