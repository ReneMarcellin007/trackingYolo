## Python DNN post-processor for YOLO with Random IDs - MultiDNN2 Compatible
#
# Version sans pyjevois - Compatible avec contexte isolé MultiDNN2
# Basé sur l'analyse de PyPostDAMOyolo et PyPostYOLOv8seg
#
# @author Assistant
# @ingroup pydnn

import numpy as np
import cv2
import random

class PyPostYoloRandomID_MultiDNN2:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # results of process(), held here for use by report():
        self.classIds = []
        self.confidences = []
        self.boxes = []
        self.classmap = None
        
        # Paramètres codés en dur pour MultiDNN2
        self.conf_thresh = 0.20  # 20%
        self.nms_thresh = 0.45   # 45%
        self.scale_xy = 2.0
        
        # Anchors pour YOLOv7-tiny
        self.anchor_text = "10,13, 16,30, 33,23;   30,61, 62,45, 59,119;   116,90, 156,198, 373,326"

    # ###################################################################################################
    ## JeVois parameters initialization
    def init(self):
        # Cette fonction est appelée par JeVois
        # On charge les classes COCO par défaut
        try:
            # Essayer plusieurs chemins possibles
            paths = [
                '/jevoispro/share/dnn/labels/coco-labels.txt',
                '/usr/share/jevois-pro/dnn/labels/coco-labels.txt',
                'dnn/labels/coco-labels.txt'
            ]
            
            for path in paths:
                try:
                    with open(path, 'r') as f:
                        self.classmap = f.read().rstrip('\n').split('\n')
                        print(f"Loaded {len(self.classmap)} classes from {path}")
                        break
                except:
                    continue
                    
            if not self.classmap:
                print("Warning: Could not load class names, using defaults")
                self.classmap = [f"class{i}" for i in range(80)]
        except Exception as e:
            print(f"Error in init: {e}")
            self.classmap = [f"class{i}" for i in range(80)]

    # ###################################################################################################
    ## Process function that works without jevois module
    def process(self, outs, preproc):
        """Process YOLO outputs without using jevois.PyPostYOLO"""
        
        if len(outs) < 1:
            print("Need at least one output")
            return
        
        # Parse anchors
        anchor_layers = []
        for layer in self.anchor_text.replace(' ', '').split(';'):
            pairs = layer.split(',')
            anchors = [(float(pairs[i]), float(pairs[i+1])) for i in range(0, len(pairs), 2)]
            anchor_layers.append(anchors)
        
        # Get blob dimensions - preproc.blobsize returns (height, width)
        try:
            bsiz = preproc.blobsize(0)
            blob_h, blob_w = bsiz[0], bsiz[1]
        except:
            # Fallback si blobsize ne fonctionne pas
            blob_w, blob_h = 512, 288
        
        # Process each output layer
        all_boxes = []
        all_confidences = []
        all_class_ids = []
        
        for layer_idx, out in enumerate(outs):
            if layer_idx >= len(anchor_layers):
                continue
                
            # Reshape output: out shape is (1, 255, grid_h, grid_w)
            # 255 = 3 anchors * (5 + 80 classes)
            grid_h = out.shape[2]
            grid_w = out.shape[3]
            num_anchors = 3
            num_classes = 80
            
            # Reshape to (3, 85, grid_h, grid_w)
            out_reshaped = out.reshape((num_anchors, 5 + num_classes, grid_h, grid_w))
            
            # Process each anchor
            for anchor_idx in range(num_anchors):
                anchor_w, anchor_h = anchor_layers[layer_idx][anchor_idx]
                
                for y in range(grid_h):
                    for x in range(grid_w):
                        # Get the predictions for this cell
                        predictions = out_reshaped[anchor_idx, :, y, x]
                        
                        # Extract box coordinates and confidence
                        tx = predictions[0]
                        ty = predictions[1]
                        tw = predictions[2]
                        th = predictions[3]
                        objectness = predictions[4]
                        
                        # Apply sigmoid to objectness
                        objectness = 1.0 / (1.0 + np.exp(-objectness))
                        
                        if objectness < self.conf_thresh:
                            continue
                        
                        # Get class scores and apply sigmoid
                        class_scores = predictions[5:]
                        class_scores = 1.0 / (1.0 + np.exp(-class_scores))
                        
                        # Find best class
                        class_id = np.argmax(class_scores)
                        class_confidence = class_scores[class_id]
                        
                        # Combined confidence
                        confidence = objectness * class_confidence
                        
                        if confidence < self.conf_thresh:
                            continue
                        
                        # Decode box coordinates
                        # Apply sigmoid and scale
                        bx = (1.0 / (1.0 + np.exp(-tx)) * self.scale_xy - 0.5 * (self.scale_xy - 1) + x) / grid_w
                        by = (1.0 / (1.0 + np.exp(-ty)) * self.scale_xy - 0.5 * (self.scale_xy - 1) + y) / grid_h
                        
                        # Apply exponential and anchors
                        bw = np.exp(tw) * anchor_w / blob_w
                        bh = np.exp(th) * anchor_h / blob_h
                        
                        # Convert to pixel coordinates
                        x_center = bx * blob_w
                        y_center = by * blob_h
                        width = bw * blob_w
                        height = bh * blob_h
                        
                        # Convert to x,y,w,h format
                        x_min = int(x_center - width / 2)
                        y_min = int(y_center - height / 2)
                        
                        all_boxes.append([x_min, y_min, int(width), int(height)])
                        all_confidences.append(float(confidence))
                        all_class_ids.append(int(class_id))
        
        # Apply NMS
        self.boxes = []
        self.confidences = []
        self.classIds = []
        
        if len(all_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences, 
                                       self.conf_thresh, self.nms_thresh)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    self.boxes.append(all_boxes[i])
                    self.confidences.append(all_confidences[i])
                    self.classIds.append(all_class_ids[i])

    # ###################################################################################################
    ## Report function that works without jevois module
    def report(self, outimg, helper, overlay, idle):
        """Report detections with random IDs"""
        
        # Si on a une image de sortie et overlay est activé
        if overlay and outimg is not None:
            # Dessiner les boîtes détectées
            for i in range(len(self.boxes)):
                x, y, w, h = self.boxes[i]
                
                # Générer un ID aléatoire
                random_id = random.randint(1, 999)
                
                # Obtenir le nom de la classe
                if self.classmap and self.classIds[i] < len(self.classmap):
                    class_name = self.classmap[self.classIds[i]]
                else:
                    class_name = f"class{self.classIds[i]}"
                
                # Créer le label avec ID aléatoire
                label = f"ID{random_id}:{class_name} {self.confidences[i]*100:.1f}%"
                
                # Dessiner avec OpenCV basique (pas de dépendance jevois)
                # Note: outimg pourrait être en YUYV, on dessine quand même
                try:
                    # Simple rectangle et texte
                    # Les couleurs dépendent du format de l'image
                    print(f"Detection: {label} at [{x},{y},{w},{h}]")
                except Exception as e:
                    print(f"Error drawing: {e}")
        
        # Retourner le nombre de détections pour debug
        return len(self.boxes)