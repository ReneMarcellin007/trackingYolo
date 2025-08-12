import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
import random

## YOLO post-processor with Random IDs - MultiDNN2 Safe Version
#
# @author Laurent Itti
# 
# @ingroup pydnn

class PyPostYoloRandomID:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        self.classIds = []
        self.confidences = []
        self.boxes = []
        self.classmap = None
        
        # Essayer de créer PyPostYOLO (fonctionne en DNN normal)
        self.yolopp = None
        try:
            self.yolopp = jevois.PyPostYOLO()
        except:
            # Normal en MultiDNN2, on réessayera dans init()
            pass
        
    # ###################################################################################################
    ## Initialisation des paramètres
    def init(self):
        pc = jevois.ParameterCategory("DNN Post-Processing Options", "")
        
        # Deuxième tentative pour MultiDNN2
        if self.yolopp is None:
            try:
                self.yolopp = jevois.PyPostYOLO()
            except:
                # Si ça échoue encore, on ne peut pas utiliser PyPostYOLO
                pass
        
        self.classoffset = jevois.Parameter(self, 'classoffset', 'int',
                        "Offset to apply to class indices",
                        0, pc)

        self.classes = jevois.Parameter(self, 'classes', 'str',
                        "Path to text file with names of object classes",
                        'dnn/labels/coco-labels.txt', pc)
        self.classes.setCallback(self.loadClasses)

        self.detecttype = jevois.Parameter(self, 'detecttype', 'str',
                        "Type of detection output format",
                        'RAWYOLO', pc)
        self.detecttype.setCallback(self.setDetectType)

        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                        "Classification threshold, in percent confidence",
                        20.0, pc)

        self.dthresh = jevois.Parameter(self, 'dthresh', 'float', 
                        "Detection box threshold (for RAWYOLO only), in percent confidence",
                        15.0, pc)

        self.nms = jevois.Parameter(self, 'nms', 'float',
                        "Non-maximum suppression intersection-over-union threshold in percent",
                        45.0, pc)

        self.maxnbox = jevois.Parameter(self, 'maxnbox', 'int',
                        "Max number of detection boxes to output",
                        500, pc)

        self.sigmoid = jevois.Parameter(self, 'sigmoid', 'bool',
                        "Apply sigmoid to raw YOLO outputs",
                        False, pc)
        
        # Charger les classes
        try:
            self.loadClasses('dnn/labels/coco-labels.txt')
        except:
            pass

    # ###################################################################################################
    ## Geler les paramètres
    def freeze(self, doit):
        self.classes.freeze(doit)
        self.detecttype.freeze(doit)
        if self.yolopp is not None:
            self.yolopp.freeze(doit)

    # ###################################################################################################
    ## Charger les classes
    def loadClasses(self, filename):
        if filename:
            try:
                f = open(pyjevois.share + '/' + filename, 'rt')
                self.classmap = f.read().rstrip('\n').split('\n')
                f.close()
                jevois.LINFO(f"Loaded {len(self.classmap)} classes")
            except:
                # Classes COCO par défaut
                self.classmap = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
                                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
                                'toothbrush']

    # ###################################################################################################
    ## Callback type de détection
    def setDetectType(self, dt):
        if dt != 'RAWYOLO':
            jevois.LFATAL(f"Invalid detecttype {dt}")

    # ###################################################################################################
    ## Traitement principal - NE PAS créer PyPostYOLO ici !
    def process(self, outs, preproc):
        if len(outs) == 0:
            jevois.LFATAL("No outputs received")
        
        # Si PyPostYOLO n'est pas disponible, on ne peut pas traiter
        if self.yolopp is None:
            # En MultiDNN2, cela peut être normal
            # On retourne simplement sans détection
            return
        
        # S'assurer que les classes sont chargées
        if self.classmap is None:
            self.loadClasses('dnn/labels/coco-labels.txt')
        
        # Effacer les anciens résultats
        self.classIds.clear()
        self.confidences.clear()
        self.boxes.clear()
        
        # Dimensions
        self.imagew, self.imageh = preproc.imagesize()
        bw, bh = preproc.blobsize(0)

        # Nombre de classes
        num_classes = len(self.classmap) if self.classmap else 80

        # Traiter avec PyPostYOLO
        try:
            classids, confs, boxes = self.yolopp.yolo(outs,
                                                      num_classes,
                                                      self.dthresh.get() * 0.01,
                                                      self.cthresh.get() * 0.01,
                                                      bw, bh,
                                                      self.classoffset.get(),
                                                      self.maxnbox.get(),
                                                      self.sigmoid.get())
            
            # NMS
            indices = cv2.dnn.NMSBoxes(boxes, confs, self.cthresh.get() * 0.01, self.nms.get() * 0.01)

            # Traiter les détections
            for i in indices:
                x, y, w, h = boxes[i]

                # Limiter et convertir
                x1 = min(bw - 1, max(0, x))
                x2 = min(bw - 1, max(0, x + w))
                y1 = min(bh - 1, max(0, y))
                y2 = min(bh - 1, max(0, y + h))

                x1, y1 = preproc.b2i(x1, y1, 0)
                x2, y2 = preproc.b2i(x2, y2, 0)
                
                self.boxes.append([x1, y1, x2, y2])
            
            self.classIds = [classids[i] for i in indices]
            self.confidences = [confs[i] for i in indices]
            
        except Exception as e:
            # En cas d'erreur, pas de détection
            pass
        
    # ###################################################################################################
    ## Label avec ID aléatoire
    def getLabel(self, id, conf):
        if self.classmap is None:
            self.loadClasses('dnn/labels/coco-labels.txt')
            
        if self.classmap and id >= 0 and id < len(self.classmap): 
            categ = self.classmap[id]
        else: 
            categ = f'class{id}'
        
        color = jevois.stringToRGBA(categ, 255)
        
        # ID aléatoire
        random_id = random.randint(1, 999)
        label = "ID%d:%s: %.2f" % (random_id, categ, conf * 100.0)
        
        return (label, color & 0xffffffff)
    
    # ###################################################################################################
    ## Affichage
    def report(self, outimg, helper, overlay, idle):
        if not overlay:
            return
            
        # Mode legacy
        if outimg is not None:
            for i in range(len(self.classIds)):
                label, color = self.getLabel(self.classIds[i], self.confidences[i])
                x1, y1, x2, y2 = self.boxes[i]
                jevois.drawRect(outimg, x1, y1, x2-x1+1, y2-y1+1, 2, jevois.YUYV.LightGreen)
                jevois.writeText(outimg, label, x1 + 6, y1 + 2, jevois.YUYV.LightGreen, jevois.Font.Font10x20)

        # Mode JeVois-Pro
        if helper is not None:
            for i in range(len(self.classIds)):
                label, color = self.getLabel(self.classIds[i], self.confidences[i])
                x1, y1, x2, y2 = self.boxes[i]
                helper.drawRect(x1, y1, x2, y2, color, True)
                helper.drawText(x1 + 3, y1 + 3, label, color)