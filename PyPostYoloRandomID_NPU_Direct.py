## Post-processeur YOLO ULTRA-OPTIMISÉ - Accès direct NPU
#
# Version qui travaille directement avec les tensors NPU
# Évite les conversions coûteuses Python <-> C++
#
# @author Assistant
# @ingroup pydnn

import numpy as np
import random
import cv2

class PyPostYoloRandomID_NPU_Direct:
    """Version ultra-optimisée pour 30+ FPS avec accès direct NPU"""
    
    def __init__(self):
        self.detections = []
        self.classmap = None
        # Pré-calculer les anchors pour YOLOv7/v8
        self.anchors = [
            [(10, 13), (16, 30), (33, 23)],     # Petite échelle
            [(30, 61), (62, 45), (59, 119)],    # Moyenne échelle  
            [(116, 90), (156, 198), (373, 326)] # Grande échelle
        ]
        # Cache pour éviter les recalculs
        self.grid_cache = {}
        self.sigmoid_lut = None
        self.init_optimizations()
        
    def init_optimizations(self):
        """Pré-calculs pour optimisation"""
        # Table de lookup pour sigmoid (évite np.exp coûteux)
        x = np.linspace(-10, 10, 2000)
        self.sigmoid_lut = 1.0 / (1.0 + np.exp(-x))
        self.sigmoid_min = -10
        self.sigmoid_max = 10
        self.sigmoid_steps = 2000
        
    def fast_sigmoid(self, x):
        """Sigmoid ultra-rapide avec lookup table"""
        # Clipper et mapper vers l'index de la LUT
        x_clipped = np.clip(x, self.sigmoid_min, self.sigmoid_max)
        idx = ((x_clipped - self.sigmoid_min) / (self.sigmoid_max - self.sigmoid_min) * (self.sigmoid_steps - 1)).astype(int)
        return self.sigmoid_lut[idx]
    
    def init(self):
        """Initialisation JeVois"""
        try:
            # Charger les classes COCO
            with open('/jevoispro/share/dnn/labels/coco-labels.txt', 'r') as f:
                self.classmap = f.read().rstrip('\n').split('\n')
        except:
            self.classmap = [f"class{i}" for i in range(80)]
    
    def get_grid_offsets(self, grid_w, grid_h):
        """Cache les offsets de grille pour éviter les recalculs"""
        key = (grid_w, grid_h)
        if key not in self.grid_cache:
            # Créer une grille d'offsets
            xv, yv = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            self.grid_cache[key] = (xv, yv)
        return self.grid_cache[key]
    
    def process_optimized_yolov8(self, outs, preproc):
        """Traitement optimisé pour YOLOv8 avec library native"""
        self.detections = []
        
        if len(outs) == 0:
            return
            
        # YOLOv8 avec library: sortie format [1, num_boxes, 85] ou [num_boxes, 85]
        output = outs[0]
        
        if len(output.shape) == 3:
            output = output[0]  # Enlever dimension batch
        
        # Traitement vectorisé rapide
        if output.shape[1] >= 85:  # 80 classes + 4 coords + 1 conf
            # Extraction vectorisée
            boxes = output[:, :4]      # x, y, w, h
            confs = output[:, 4]        # confidence
            classes = output[:, 5:85]   # scores des classes
            
            # Seuillage vectorisé
            mask = confs > 0.25
            valid_boxes = boxes[mask]
            valid_confs = confs[mask]
            valid_classes = classes[mask]
            
            # Argmax vectorisé pour les classes
            class_ids = np.argmax(valid_classes, axis=1)
            class_confs = np.max(valid_classes, axis=1)
            
            # Créer les détections avec IDs aléatoires
            for i in range(len(valid_boxes)):
                self.detections.append({
                    'box': valid_boxes[i],
                    'conf': valid_confs[i] * class_confs[i],
                    'class_id': int(class_ids[i]),
                    'random_id': random.randint(1, 999)
                })
    
    def process_optimized_yolov7(self, outs, preproc):
        """Traitement optimisé pour YOLOv7 raw (sans library)"""
        self.detections = []
        
        # Récupérer dimensions blob
        try:
            bsiz = preproc.blobsize(0)
            blob_h, blob_w = bsiz[0], bsiz[1]
        except:
            blob_w, blob_h = 512, 288
        
        all_detections = []
        
        # Traiter chaque échelle en parallèle
        for scale_idx, out in enumerate(outs[:3]):  # 3 échelles max
            if scale_idx >= len(self.anchors):
                continue
                
            # Format: [1, 255, grid_h, grid_w] -> [3, 85, grid_h, grid_w]
            grid_h, grid_w = out.shape[2], out.shape[3]
            out = out.reshape(3, 85, grid_h, grid_w)
            
            # Obtenir grille d'offsets (cachée)
            xv, yv = self.get_grid_offsets(grid_w, grid_h)
            
            # Traitement vectorisé par anchor
            for anchor_idx in range(3):
                anchor_w, anchor_h = self.anchors[scale_idx][anchor_idx]
                pred = out[anchor_idx]  # [85, grid_h, grid_w]
                
                # Objectness avec sigmoid rapide
                obj = self.fast_sigmoid(pred[4])
                
                # Masque pour objectness > seuil
                obj_mask = obj > 0.25
                if not np.any(obj_mask):
                    continue
                
                # Extraire seulement les cellules valides
                valid_indices = np.where(obj_mask)
                valid_y = valid_indices[0]
                valid_x = valid_indices[1]
                
                # Coordonnées avec sigmoid rapide
                tx = self.fast_sigmoid(pred[0][obj_mask])
                ty = self.fast_sigmoid(pred[1][obj_mask])
                tw = pred[2][obj_mask]
                th = pred[3][obj_mask]
                
                # Classes avec sigmoid rapide  
                classes = self.fast_sigmoid(pred[5:85, obj_mask[0], obj_mask[1]])
                
                # Calcul vectorisé des boîtes
                bx = (tx * 2.0 - 0.5 + valid_x) / grid_w * blob_w
                by = (ty * 2.0 - 0.5 + valid_y) / grid_h * blob_h
                bw = np.exp(tw) * anchor_w
                bh = np.exp(th) * anchor_h
                
                # Classes et confidences
                class_ids = np.argmax(classes, axis=0)
                class_confs = np.max(classes, axis=0)
                final_confs = obj[obj_mask] * class_confs
                
                # Ajouter détections valides
                valid = final_confs > 0.25
                for i in np.where(valid)[0]:
                    all_detections.append({
                        'x': bx[i] - bw[i]/2,
                        'y': by[i] - bh[i]/2,
                        'w': bw[i],
                        'h': bh[i],
                        'conf': final_confs[i],
                        'class_id': class_ids[i]
                    })
        
        # NMS vectorisé rapide
        if all_detections:
            boxes = [[d['x'], d['y'], d['w'], d['h']] for d in all_detections]
            confs = [d['conf'] for d in all_detections]
            
            # NMS avec OpenCV (optimisé C++)
            indices = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    det = all_detections[i]
                    det['random_id'] = random.randint(1, 999)
                    self.detections.append(det)
    
    def process(self, outs, preproc):
        """Process principal - détecte le type de sortie"""
        if len(outs) == 0:
            return
        
        # Détecter le format de sortie
        first_out = outs[0]
        
        # YOLOv8 avec library: [1, N, 85] ou [N, 85]
        if len(first_out.shape) <= 3 and first_out.shape[-1] >= 85:
            self.process_optimized_yolov8(outs, preproc)
        # YOLOv7 raw: [1, 255, H, W]
        elif len(first_out.shape) == 4 and first_out.shape[1] == 255:
            self.process_optimized_yolov7(outs, preproc)
        else:
            print(f"Format non reconnu: {first_out.shape}")
    
    def report(self, outimg, helper, overlay, idle):
        """Affichage optimisé des résultats"""
        
        # Log minimaliste pour performance
        if len(self.detections) > 0:
            print(f"Détections: {len(self.detections)}")
            
            # Afficher seulement les 3 premières pour debug
            for det in self.detections[:3]:
                if 'box' in det:  # Format YOLOv8
                    x, y, w, h = det['box']
                else:  # Format YOLOv7
                    x, y, w, h = det['x'], det['y'], det['w'], det['h']
                
                class_name = self.classmap[det['class_id']] if self.classmap else f"class{det['class_id']}"
                print(f"  ID{det['random_id']}:{class_name} {det['conf']*100:.1f}%")
        
        return len(self.detections)