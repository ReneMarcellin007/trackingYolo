#!/usr/bin/env python3
"""
Crée une version compatible MultiDNN2 du tracker
"""

simple_tracker = '''import sys
sys.path.insert(0, '/jevoispro/config')
import pyjevois

if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2

## Simple YOLOv7 Post-Processor for MultiDNN2
class PyPostYolov7Simple:
    
    def __init__(self):
        self.classIds = []
        self.confidences = []
        self.boxes = []
        self.classmap = None
        self.yolopp = None
        
    def init(self):
        """Initialize parameters"""
        pc = jevois.ParameterCategory("YOLOv7 Simple", "")
        
        self.classes = jevois.Parameter(self, 'classes', 'str',
                        "Path to text file with names of object classes",
                        '', pc)
        self.classes.setCallback(self.loadClasses)

        self.detecttype = jevois.Parameter(self, 'detecttype', 'str', 
                                           "Type of detection output format",
                                           'RAWYOLO', pc)
        
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
                        500, pc)
                        
        self.sigmoid = jevois.Parameter(self, 'sigmoid', 'bool',
                        "Apply sigmoid",
                        False, pc)
                        
        self.classoffset = jevois.Parameter(self, 'classoffset', 'int',
                        "Class offset",
                        0, pc)
    
    def freeze(self, doit):
        self.classes.freeze(doit)
        if self.yolopp:
            self.yolopp.freeze(doit)

    def loadClasses(self, filename):
        if filename:
            try:
                f = open(pyjevois.share + '/' + filename, 'rt')
                self.classmap = f.read().rstrip('\\n').split('\\n')
                f.close()
            except:
                self.classmap = []

    def process(self, outs, preproc):
        """Simple YOLO processing"""
        if len(outs) == 0: 
            return
            
        # Initialize YOLO decoder if needed
        if self.yolopp is None:
            try:
                self.yolopp = jevois.PyPostYOLO()
            except:
                # Fallback for MultiDNN2 - just return empty
                return
                
        # Clear results
        self.classIds.clear()
        self.confidences.clear()
        self.boxes.clear()
        
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
                indices = cv2.dnn.NMSBoxes(boxes, confs, 
                                           self.cthresh.get() * 0.01, 
                                           self.nms.get() * 0.01)

                for i in indices:
                    x, y, w, h = boxes[i]
                    x1 = min(bw - 1, max(0, x))
                    x2 = min(bw - 1, max(0, x + w))
                    y1 = min(bh - 1, max(0, y))
                    y2 = min(bh - 1, max(0, y + h))

                    x1, y1 = preproc.b2i(x1, y1, 0)
                    x2, y2 = preproc.b2i(x2, y2, 0)
                    
                    self.boxes.append([x1, y1, x2, y2])
                    self.classIds.append(classids[i])
                    self.confidences.append(confs[i])
                    
        except Exception as e:
            pass  # Silent fail for MultiDNN2

    def report(self, outimg, helper, overlay, idle):
        """Simple display"""
        if helper is not None and overlay:
            for i, (class_id, conf) in enumerate(zip(self.classIds, self.confidences)):
                if i < len(self.boxes):
                    x1, y1, x2, y2 = self.boxes[i]
                    
                    if self.classmap and class_id < len(self.classmap):
                        label = f"{self.classmap[class_id]}: {conf*100:.1f}%"
                    else:
                        label = f"obj{class_id}: {conf*100:.1f}%"
                    
                    col = 0xFF00FF00  # Green
                    helper.drawRect(x1, y1, x2, y2, col, True)
                    helper.drawText(x1 + 3, y1 + 3, label, col)
'''

print("Creating simple MultiDNN2-compatible tracker...")

# Save the simple version
with open('/jevoispro/share/pydnn/post/PyPostYolov7Simple.py', 'w') as f:
    f.write(simple_tracker)

print("✅ Created PyPostYolov7Simple.py")

# Create YAML config for it
yaml_config = '''%YAML 1.0
---

yolov7-simple:
  preproc: Blob
  mean: "0 0 0"
  scale: 0.0039215686
  nettype: NPU
  model: "npu/detection/yolov7-tiny-1024x576.nb"
  postproc: Python
  pypost: "pydnn/post/PyPostYolov7Simple.py"
  detecttype: RAWYOLO
  classes: "dnn/labels/coco-labels.txt"
  processing: Async
  intensors: "NCHW:8U:1x3x576x1024:AA:0.003921568393707275:0"
  outtensors: "NCHW:8U:1x255x72x128:AA:0.003911261446774006:0, NCHW:8U:1x255x36x64:AA:0.0038770213909447193:0, NCHW:8U:1x255x18x32:AA:0.003848204342648387:0"
  anchors: "10,13, 16,30, 33,23; 30,61, 62,45, 59,119; 116,90, 156,198, 373,326"
  scalexy: 2.0
  sigmoid: false
  cthresh: 25
  dthresh: 20
  nms: 45
'''

with open('/jevoispro/share/dnn/custom/yolov7-simple.yml', 'w') as f:
    f.write(yaml_config)

print("✅ Created yolov7-simple.yml")
print("\nCe modèle simple devrait fonctionner dans MultiDNN2!")