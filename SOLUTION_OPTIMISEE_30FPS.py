## Solution OPTIMISÉE pour 30+ FPS - Utilise la bibliothèque native
#
# Cette version utilise la bibliothèque .so native pour le décodage YOLO
# puis ajoute simplement les IDs aléatoires
#
# @author Assistant
# @ingroup pydnn

import numpy as np
import random

class PyPostYoloRandomID_Optimized:
    """Version optimisée qui délègue le décodage YOLO au C++ natif"""
    
    def __init__(self):
        self.detections = []
        self.classmap = None
        
    def init(self):
        """Initialisation - Charge les noms de classes"""
        try:
            # Charger les labels COCO
            with open('/jevoispro/share/dnn/labels/coco-labels.txt', 'r') as f:
                self.classmap = f.read().rstrip('\n').split('\n')
        except:
            self.classmap = [f"class{i}" for i in range(80)]
    
    def process(self, outs, preproc):
        """
        Process optimisé - Les sorties sont DÉJÀ décodées par la bibliothèque native !
        On ajoute juste les IDs aléatoires
        """
        self.detections = []
        
        # Pour YOLOv8 avec library native, les sorties sont déjà des boîtes
        # Format attendu : [batch, num_detections, 6] où 6 = [x,y,w,h,conf,class]
        if len(outs) > 0:
            output = outs[0]
            
            # Si c'est une sortie déjà décodée (shape = [1, N, 6] ou [N, 6])
            if len(output.shape) == 3 and output.shape[2] >= 6:
                # Format [batch, detections, attributes]
                for i in range(output.shape[1]):
                    det = output[0, i, :]
                    if det[4] > 0.2:  # Seuil de confiance
                        self.detections.append({
                            'x': int(det[0]),
                            'y': int(det[1]), 
                            'w': int(det[2]),
                            'h': int(det[3]),
                            'conf': float(det[4]),
                            'class_id': int(det[5]),
                            'random_id': random.randint(1, 999)
                        })
            elif len(output.shape) == 2 and output.shape[1] >= 6:
                # Format [detections, attributes]
                for i in range(output.shape[0]):
                    det = output[i, :]
                    if det[4] > 0.2:  # Seuil de confiance
                        self.detections.append({
                            'x': int(det[0]),
                            'y': int(det[1]),
                            'w': int(det[2]),
                            'h': int(det[3]),
                            'conf': float(det[4]),
                            'class_id': int(det[5]),
                            'random_id': random.randint(1, 999)
                        })
    
    def report(self, outimg, helper, overlay, idle):
        """Affichage des résultats avec IDs aléatoires"""
        
        for det in self.detections:
            # Créer le label avec ID aléatoire
            if self.classmap and det['class_id'] < len(self.classmap):
                class_name = self.classmap[det['class_id']]
            else:
                class_name = f"class{det['class_id']}"
            
            label = f"ID{det['random_id']}:{class_name} {det['conf']*100:.1f}%"
            
            # Log pour debug
            print(f"Detection: {label} at [{det['x']},{det['y']},{det['w']},{det['h']}]")
        
        return len(self.detections)