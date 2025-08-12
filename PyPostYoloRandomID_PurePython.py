import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
import random

## Python DNN post-processor for YOLO with Random IDs - Pure Python Version
#
# Version compatible avec MultiDNN2 - Sans PyPostYOLO
# Implémente le décodage YOLO en Python pur comme PyPostDAMOyolo
#
# @author Assistant
# @ingroup pydnn
class PyPostYoloRandomID_PurePython:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # results of process(), held here for use by report():
        self.classIds = []
        self.confidences = []
        self.boxes = []
        self.classmap = None

    # ###################################################################################################
    ## JeVois parameters initialization
    def init(self):
        pc = jevois.ParameterCategory("DNN Post-Processing Options", "")
        
        self.classes = jevois.Parameter(self, 'classes', 'str',
                       "Path to text file with names of object classes",
                       'dnn/labels/coco-labels.txt', pc)
        self.classes.setCallback(self.loadClasses)
        
        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                      "Detection threshold in percent",
                      20.0, pc)
        
        self.nms = jevois.Parameter(self, 'nms', 'float',
                   "Non-maximum suppression intersection-over-union threshold in percent",
                   45.0, pc)
        
        self.anchors = jevois.Parameter(self, 'anchors', 'str',
                      "Anchor boxes, usually should not be changed",
                      "10,13, 16,30, 33,23;   30,61, 62,45, 59,119;   116,90, 156,198, 373,326", pc)
        
        self.scalexy = jevois.Parameter(self, 'scalexy', 'float',
                      "Non-linear coordinate box scaling, usually should not be changed",
                      2.0, pc)

    # ###################################################################################################
    ## Load class names
    def loadClasses(self, filename):
        if filename:
            with open(jevois.share + '/' + filename, 'r') as f:
                self.classmap = f.read().rstrip('\n').split('\n')

    # ###################################################################################################
    ## Decode YOLO raw outputs - Pure Python implementation
    def decode_yolo_output(self, output, blob_w, blob_h, anchor_layer):
        """Décoder une sortie YOLO en boîtes et scores"""
        grid_h, grid_w = output.shape[2], output.shape[3]
        num_anchors = len(anchor_layer)
        num_classes = output.shape[1] // num_anchors - 5
        
        output = output.reshape((num_anchors, 5 + num_classes, grid_h, grid_w))
        output = output.transpose(0, 2, 3, 1)  # anchors, grid_h, grid_w, (5+classes)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for anchor_idx in range(num_anchors):
            for y in range(grid_h):
                for x in range(grid_w):
                    data = output[anchor_idx, y, x]
                    
                    # Coordonnées de la boîte
                    tx, ty, tw, th = data[0:4]
                    objectness = data[4]
                    
                    # Sigmoid pour objectness
                    objectness = 1.0 / (1.0 + np.exp(-objectness))
                    
                    if objectness < self.cthresh.get() / 100.0:
                        continue
                    
                    # Scores des classes
                    class_scores = data[5:]
                    class_scores = 1.0 / (1.0 + np.exp(-class_scores))  # Sigmoid
                    class_id = np.argmax(class_scores)
                    confidence = objectness * class_scores[class_id]
                    
                    if confidence < self.cthresh.get() / 100.0:
                        continue
                    
                    # Décodage des coordonnées avec anchors
                    anchor_w, anchor_h = anchor_layer[anchor_idx]
                    scale = self.scalexy.get()
                    
                    # Sigmoid et scaling pour x,y
                    bx = (1.0 / (1.0 + np.exp(-tx)) * scale - 0.5 * (scale - 1) + x) * blob_w / grid_w
                    by = (1.0 / (1.0 + np.exp(-ty)) * scale - 0.5 * (scale - 1) + y) * blob_h / grid_h
                    
                    # Exponentielle pour largeur/hauteur
                    bw = np.exp(tw) * anchor_w * blob_w / grid_w
                    bh = np.exp(th) * anchor_h * blob_h / grid_h
                    
                    # Convertir en x,y,w,h
                    x_min = int(bx - bw / 2)
                    y_min = int(by - bh / 2)
                    
                    boxes.append([x_min, y_min, int(bw), int(bh)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
        
        return boxes, confidences, class_ids

    # ###################################################################################################
    ## Process outputs
    def process(self, outs, preproc):
        if len(outs) < 1:
            jevois.LERROR("Need at least one output")
            return
        
        # Parse anchors
        anchor_text = self.anchors.get().replace(' ', '')
        anchor_layers = []
        for layer in anchor_text.split(';'):
            pairs = layer.split(',')
            anchors = [(float(pairs[i]), float(pairs[i+1])) for i in range(0, len(pairs), 2)]
            anchor_layers.append(anchors)
        
        # Get blob dimensions
        bsiz = preproc.blobsize(0)
        blob_w, blob_h = bsiz[1], bsiz[0]
        
        # Decode each output layer
        all_boxes = []
        all_confidences = []
        all_class_ids = []
        
        for i, out in enumerate(outs):
            if i < len(anchor_layers):
                boxes, confs, ids = self.decode_yolo_output(out, blob_w, blob_h, anchor_layers[i])
                all_boxes.extend(boxes)
                all_confidences.extend(confs)
                all_class_ids.extend(ids)
        
        # Apply NMS
        if len(all_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences, 
                                       self.cthresh.get() / 100.0, 
                                       self.nms.get() / 100.0)
            
            self.boxes = []
            self.confidences = []
            self.classIds = []
            
            if len(indices) > 0:
                for i in indices.flatten():
                    self.boxes.append(all_boxes[i])
                    self.confidences.append(all_confidences[i])
                    self.classIds.append(all_class_ids[i])
        else:
            self.boxes = []
            self.confidences = []
            self.classIds = []

    # ###################################################################################################
    ## Report results
    def report(self, outimg, helper, overlay, idle):
        if overlay and outimg is not None:
            for i in range(len(self.boxes)):
                x, y, w, h = self.boxes[i]
                
                # Générer un ID aléatoire
                random_id = random.randint(1, 999)
                
                # Nom de classe
                if self.classmap and self.classIds[i] < len(self.classmap):
                    class_name = self.classmap[self.classIds[i]]
                else:
                    class_name = f"class{self.classIds[i]}"
                
                # Label avec ID aléatoire
                label = f"ID{random_id}:{class_name} {self.confidences[i]*100:.1f}%"
                
                # Dessiner la boîte et le label
                jevois.drawRect(outimg, x, y, w, h, 2, jevois.YUYV.MedGreen)
                jevois.writeText(outimg, label, x + 3, y - 12, jevois.YUYV.MedGreen, jevois.Font.Font10x20)