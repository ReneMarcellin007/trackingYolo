## Wrapper minimal pour YOLOv8 avec IDs aléatoires
# YOLOv8 utilise déjà un décodeur C++ natif, on ajoute juste les IDs

import numpy as np
import random

class PyPostYolov8RandomID:
    def __init__(self):
        self.detections = []
    
    def init(self):
        pass
    
    def process(self, outs, preproc):
        # YOLOv8 retourne déjà les détections décodées
        # Format: [x, y, w, h, confidence, class_id]
        self.detections = []
        
        if len(outs) > 0:
            out = outs[0]
            # Parcourir les détections
            for i in range(out.shape[0]):
                detection = out[i]
                # Ajouter un ID aléatoire
                random_id = random.randint(1, 999)
                self.detections.append({
                    'box': detection[:4],
                    'confidence': detection[4],
                    'class_id': int(detection[5]),
                    'random_id': random_id
                })
    
    def report(self, outimg, helper, overlay, idle):
        if overlay and outimg:
            for det in self.detections:
                x, y, w, h = det['box']
                label = f"ID{det['random_id']}:class{det['class_id']} {det['confidence']*100:.1f}%"
                print(f"Detection: {label}")
        return len(self.detections)
